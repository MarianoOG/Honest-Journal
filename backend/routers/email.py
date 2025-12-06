"""
Email router for sending feedback and notifications.
"""
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from sqlmodel import Session, select, func
from config import Settings, logger
from models import User, Reflection

settings = Settings()
router = APIRouter(prefix="/email", tags=["email"])


def get_database_engine():
    """Get the database engine from the main app context"""
    from fastapi_app import database_engine
    return database_engine


class FeedbackRequest(BaseModel):
    """Schema for feedback submission."""
    issue_type: str
    description: str
    session_info: dict = {}


@router.post("/send-feedback", status_code=status.HTTP_200_OK)
async def send_feedback(feedback: FeedbackRequest) -> dict:
    """
    Send feedback via email.

    Args:
        feedback: FeedbackRequest containing issue type, description, and optional session info

    Returns:
        dict: Success message if email sent successfully

    Raises:
        HTTPException: 400 if email configuration is missing
        HTTPException: 500 if email sending fails
    """
    # Validate email configuration
    if not settings.SENDER_EMAIL or not settings.SENDER_PASSWORD or not settings.RECIPIENT_EMAIL:
        logger.error("Email configuration is incomplete - missing SENDER_EMAIL, SENDER_PASSWORD, or RECIPIENT_EMAIL")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email service is not properly configured"
        )

    try:
        # Create message
        message = MIMEMultipart()
        message["From"] = settings.SENDER_EMAIL
        message["To"] = settings.RECIPIENT_EMAIL
        message["Subject"] = feedback.issue_type

        # Format session info
        session_info_text = ""
        if feedback.session_info:
            sorted_keys = sorted(feedback.session_info.keys())
            for key in sorted_keys:
                value = feedback.session_info[key]
                session_info_text += f"{key}: {value}\n"

        # Create email body
        email_body = f"""
Feedback Report
===============

Type: {feedback.issue_type}
Timestamp: {datetime.now().isoformat()}

Description:
{feedback.description}

---

Session Information:

{session_info_text}
        """

        message.attach(MIMEText(email_body, "plain"))

        with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:
            server.starttls()  # Upgrade to secure connection
            server.login(settings.SENDER_EMAIL, settings.SENDER_PASSWORD)
            server.send_message(message)

        logger.info(f"Feedback email sent successfully for issue type: {feedback.issue_type}")
        return {"message": "Feedback submitted successfully"}

    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP authentication failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to authenticate with email service"
        )
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error while sending email: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send email"
        )
    except Exception as e:
        logger.error(f"Unexpected error sending email: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while sending email"
        )


@router.post("/send-kpi", status_code=status.HTTP_200_OK)
async def send_kpi_email() -> dict:
    """
    Calculate and send KPI metrics via email.

    Calculates the following metrics:
    - Active Journaling Rate: % of users making ≥3 entries/week
    - Average Entries Per Active User: Mean entries/week among users who made ≥1 entry
    - Weekly Active Users (WAU): % of users who made ≥1 entry this week

    Returns:
        dict: Success message if email sent successfully

    Raises:
        HTTPException: 400 if email configuration is missing
        HTTPException: 500 if email sending fails
    """
    # Validate email configuration
    if not settings.SENDER_EMAIL or not settings.SENDER_PASSWORD or not settings.RECIPIENT_EMAIL:
        logger.error("Email configuration is incomplete - missing SENDER_EMAIL, SENDER_PASSWORD, or RECIPIENT_EMAIL")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email service is not properly configured"
        )

    try:
        with Session(get_database_engine()) as session:
            # Calculate date range for this week (last 7 days)
            week_ago = datetime.now() - timedelta(days=7)

            # Get total number of users
            total_users_query = select(func.count(User.id))
            total_users = session.exec(total_users_query).one()

            # Get users with entries this week and their entry counts
            users_with_entries_query = select(
                Reflection.user_id,
                func.count(Reflection.id).label("entry_count")
            ).where(
                Reflection.created_at >= week_ago
            ).group_by(Reflection.user_id)

            users_with_entries = session.exec(users_with_entries_query).all()

            # Calculate metrics
            weekly_active_users = len(users_with_entries)
            users_with_3_plus = sum(1 for _, count in users_with_entries if count >= 3)
            total_entries_this_week = sum(count for _, count in users_with_entries)

            # Calculate percentages and averages
            wau_percentage = (weekly_active_users / total_users * 100) if total_users > 0 else 0
            active_journaling_rate = (users_with_3_plus / total_users * 100) if total_users > 0 else 0
            avg_entries_per_active_user = (total_entries_this_week / weekly_active_users) if weekly_active_users > 0 else 0

            # Create email message
            message = MIMEMultipart()
            message["From"] = settings.SENDER_EMAIL
            message["To"] = settings.RECIPIENT_EMAIL
            message["Subject"] = f"Honest Journal KPIs - Week of {datetime.now().strftime('%Y-%m-%d')}"

            # Create formatted email body
            email_body = f"""
Honest Journal - Weekly KPI Report
{'=' * 50}

Report Period: {week_ago.strftime('%Y-%m-%d')} to {datetime.now().strftime('%Y-%m-%d')}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{'=' * 50}

PRIMARY METRICS
{'=' * 50}

1. Active Journaling Rate
   Definition: % of users making ≥3 entries/week
   Value: {active_journaling_rate:.2f}%
   ({users_with_3_plus} out of {total_users} total users)

   Why It Matters: Your AI needs multiple entries to detect
   patterns and contradictions. Below 3 = insufficient data.

2. Average Entries Per Active User
   Definition: Mean entries/week among users who made ≥1 entry
   Value: {avg_entries_per_active_user:.2f} entries/week

   Why It Matters: Tracks engagement depth.
   Target: 3.5-5 entries/week for healthy usage.

3. Weekly Active Users (WAU)
   Definition: % of users who made ≥1 entry this week
   Value: {wau_percentage:.2f}%
   ({weekly_active_users} out of {total_users} total users)

   Why It Matters: Measures breadth—how many people
   are using the app at all.

{'=' * 50}

DETAILED BREAKDOWN
{'=' * 50}

Total Users: {total_users}
Users Active This Week: {weekly_active_users}
Users with ≥3 Entries: {users_with_3_plus}
Total Entries This Week: {total_entries_this_week}

{'=' * 50}

This report was automatically generated by Honest Journal.
            """

            message.attach(MIMEText(email_body, "plain"))

            # Send email via SMTP with STARTTLS (required for Cloud Run)
            # Cloud Run blocks port 465, so we use port 587 with STARTTLS
            with smtplib.SMTP(settings.SMTP_SERVER, settings.SMTP_PORT) as server:
                server.starttls()  # Upgrade to secure connection
                server.login(settings.SENDER_EMAIL, settings.SENDER_PASSWORD)
                server.send_message(message)

            logger.info("KPI email sent successfully")
            return {
                "message": "KPI email sent successfully",
                "metrics": {
                    "active_journaling_rate": f"{active_journaling_rate:.2f}%",
                    "avg_entries_per_active_user": f"{avg_entries_per_active_user:.2f}",
                    "weekly_active_users": f"{wau_percentage:.2f}%",
                    "total_users": total_users,
                    "weekly_active_count": weekly_active_users
                }
            }

    except smtplib.SMTPAuthenticationError as e:
        logger.error(f"SMTP authentication failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to authenticate with email service"
        )
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error while sending KPI email: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send KPI email"
        )
    except Exception as e:
        logger.error(f"Unexpected error sending KPI email: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred while sending KPI email: {str(e)}"
        )
