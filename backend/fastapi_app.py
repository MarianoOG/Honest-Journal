# Standard library imports
from contextlib import asynccontextmanager
import asyncio
import json

# Third-party imports
from fastapi import FastAPI
from sqlmodel import create_engine, Session, select
from google.cloud import pubsub_v1
import uvicorn

# Local application imports
from config import Settings, logger
from models import create_db_and_tables, Reflection
from routers import themes, reflections, auth, users, health, email
from ai_worker import ping_ai_worker

settings = Settings()
database_engine = None


async def keep_ai_worker_warm():
    """
    Background task that pings the AI Worker service every 5 minutes
    to prevent cold starts and keep the service responsive.
    """
    logger.info("Starting AI Worker keep-warm background task")
    while True:
        try:
            logger.info("Pinging AI Worker service to keep it warm...")
            success = ping_ai_worker()
            if success:
                logger.info("AI Worker service ping successful")
            else:
                logger.warning("AI Worker service ping failed")
            await asyncio.sleep(300)  # 5 minutes
        except asyncio.CancelledError:
            logger.info("Keep-warm task cancelled")
            break
        except Exception as e:
            logger.error(f"Unexpected error in keep-warm task: {e}")


async def listen_to_analysis_responses():
    """
    Background task that listens to Pub/Sub subscription for analysis responses
    and persists them as new reflections in the database.
    Runs blocking Pub/Sub operations in a thread pool to avoid blocking the event loop.
    """
    logger.info("Starting analysis responses listener")

    subscriber_client = pubsub_v1.SubscriberClient()
    subscription_path = f"projects/{settings.GOOGLE_CLOUD_PROJECT_ID}/subscriptions/{settings.PUBSUB_SUBSCRIPTION}"

    def pull_messages():
        """Blocking function to pull messages from Pub/Sub."""
        return subscriber_client.pull(
            request={
                "subscription": subscription_path,
                "max_messages": 10,
            },
            timeout=30,
        )

    def acknowledge_messages(ack_ids):
        """Blocking function to acknowledge messages."""
        subscriber_client.acknowledge(
            request={
                "subscription": subscription_path,
                "ack_ids": ack_ids,
            }
        )

    while True:
        try:
            # Pull messages in a thread to avoid blocking the event loop
            response = await asyncio.to_thread(pull_messages)

            if not response.received_messages:
                logger.debug("No messages received in this pull")
                await asyncio.sleep(0.1)
                continue

            logger.info(f"Received {len(response.received_messages)} messages")

            ack_ids = []
            for message in response.received_messages:
                try:
                    # Decode and parse message data
                    message_data = json.loads(message.message.data.decode('utf-8'))
                    parent_id = message_data.get("parent_id")
                    context = message_data.get("context")
                    question = message_data.get("question")

                    # DB session
                    with Session(database_engine) as session:
                        # Query parent reflection to get user_id
                        parent = session.exec(select(Reflection).where(Reflection.id == parent_id)).first()

                        # Don't ack if parent not found - will retry
                        if not parent:
                            logger.error(f"Parent reflection not found: {parent_id}")
                            continue

                        # Create new reflection
                        new_reflection = Reflection(
                            user_id=parent.user_id,
                            parent_id=parent_id,
                            context=context,
                            question=question
                        )

                        session.add(new_reflection)
                        session.commit()
                        ack_ids.append(message.ack_id)
                        logger.info(f"Created reflection {new_reflection.id} as child of {parent_id}")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message JSON: {e}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)

            # Acknowledge successfully processed messages in a thread
            if ack_ids:
                try:
                    await asyncio.to_thread(acknowledge_messages, ack_ids)
                    logger.info(f"Acknowledged {len(ack_ids)} messages")
                except Exception as e:
                    logger.error(f"Failed to acknowledge messages: {e}")

        except asyncio.CancelledError:
            logger.info("Analysis responses listener task cancelled")
            break
        except Exception as e:
            logger.error(f"Unexpected error in analysis responses listener: {e}", exc_info=True)
            await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Manage application lifespan events.
    Starts database and background tasks on startup, cleanly shuts them down on exit.
    """
    # Startup
    global database_engine
    database_engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
    create_db_and_tables(database_engine)
    logger.info("Application started with connection to the database")

    # Create and start the analysis responses listener task
    listener_task = asyncio.create_task(listen_to_analysis_responses())
    logger.info("Application started with analysis responses listener task")

    # Create and start the keep-warm background task
    keep_warm_task = asyncio.create_task(keep_ai_worker_warm())
    logger.info("Application started with AI Worker keep-warm task")

    yield

    # Shutdown
    logger.info("Shutting down, cancelling background tasks")
    keep_warm_task.cancel()
    listener_task.cancel()

    # await for keep warm task to finish
    try:
        await keep_warm_task
    except asyncio.CancelledError:
        pass

    # await for listener task to finish
    try:
        await listener_task
    except asyncio.CancelledError:
        pass

    # Close connection to database
    logger.info("Shutting down, closing connection to database")
    database_engine.dispose()

app = FastAPI(lifespan=lifespan)
app.title = "Honest Journal - Backend"
app.version = "0.0.3"

# Include routers
app.include_router(themes.router)
app.include_router(reflections.router)
app.include_router(auth.router)
app.include_router(users.router)
app.include_router(email.router)
app.include_router(health.router)


@app.get("/", 
         tags=["Root"], 
         summary="Welcome Endpoint",
         description="Returns a welcome message including the application title and version.")
def root():
    return {"message": f"Welcome to {app.title} v{app.version}"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
