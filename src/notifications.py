"""
Send automated daily trading summaries and alerts.
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, time
import pytz
from typing import List, Dict

def format_trade_recommendation(trade: Dict) -> str:
    """Format a trade recommendation for email"""
    return f"""
    {trade['action']} {trade['ticker']} @ ${trade['price']}
    Reason: {trade['reason']}
    Confidence: {trade['confidence']:.1%}
    
    Trade Instructions:
    - Entry Price: ${trade['price']}
    - Stop Loss: ${float(trade['price']) * 0.95:.2f} (5% below entry)
    - Take Profit 1: ${float(trade['price']) * 1.05:.2f} (5% gain)
    - Take Profit 2: ${float(trade['price']) * 1.10:.2f} (10% gain)
    """

def create_daily_summary(signals: Dict) -> str:
    """Create a formatted daily summary email"""
    now = datetime.now()
    
    # Start with market overview
    bull_ratio = len(signals["bullish_signals"]) / (len(signals["bearish_signals"]) or 1)
    sentiment = "Bullish" if bull_ratio > 1.2 else "Bearish" if bull_ratio < 0.8 else "Neutral"
    
    email_body = f"""
    Maritime Trading Advisor - Daily Summary
    {now.strftime('%A, %B %d, %Y')}
    
    🌊 MARKET PULSE
    Overall Sentiment: {sentiment}
    Active Signals: {len([s for s in signals["signals"] if s["confidence"] > 0.8])}
    Ports with Activity: {len(signals["port_patterns"])}
    
    📋 TODAY'S RECOMMENDATIONS
    """
    
    # Add trade recommendations
    if signals["strong_signals"]:
        for signal in signals["strong_signals"]:
            if signal["confidence"] > 0.9:
                trade = {
                    "ticker": signal["target"]["ticker"],
                    "action": "BUY" if signal["direction"] == "BULLISH" else "SELL",
                    "price": f"{signal['current_price']:.2f}",
                    "reason": f"Strong signal from {signal['port']} port activity",
                    "confidence": signal["confidence"]
                }
                email_body += "\n" + format_trade_recommendation(trade)
    else:
        email_body += "\nNo high-confidence trades recommended for today."
    
    # Add watch list
    email_body += "\n\n👀 WATCH LIST\n"
    watch_list = [s for s in signals["signals"] if 0.7 < s["confidence"] < 0.9]
    if watch_list:
        for signal in watch_list[:5]:  # Top 5 watch items
            email_body += f"""
            {signal['target']['ticker']} - {signal['direction']}
            - Watching {signal['port']} activity
            - Potential entry around ${signal['current_price']:.2f}
            """
    else:
        email_body += "No significant watch items for today."
    
    return email_body

def send_email_alert(
    recipient: str,
    subject: str,
    body: str,
    smtp_server: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str
) -> None:
    """Send an email alert"""
    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = recipient
    msg['Subject'] = subject
    
    msg.attach(MIMEText(body, 'plain'))
    
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)

def send_market_open_alert(signals: Dict, email_config: Dict) -> None:
    """Send pre-market summary before market open"""
    subject = f"Maritime Trading Advisor - Pre-Market Summary {datetime.now().strftime('%Y-%m-%d')}"
    body = create_daily_summary(signals)
    
    send_email_alert(
        recipient=email_config["recipient"],
        subject=subject,
        body=body,
        smtp_server=email_config["smtp_server"],
        smtp_port=email_config["smtp_port"],
        smtp_user=email_config["smtp_user"],
        smtp_password=email_config["smtp_password"]
    )

def send_urgent_alert(signal: Dict, email_config: Dict) -> None:
    """Send urgent alert for high-confidence signals during market hours"""
    subject = f"🚨 URGENT: New High-Confidence Signal for {signal['target']['ticker']}"
    
    body = f"""
    URGENT TRADING SIGNAL
    
    {format_trade_recommendation({
        "ticker": signal["target"]["ticker"],
        "action": "BUY" if signal["direction"] == "BULLISH" else "SELL",
        "price": f"{signal['current_price']:.2f}",
        "reason": f"Strong signal from {signal['port']} port activity",
        "confidence": signal["confidence"]
    })}
    
    This signal was generated at {datetime.now().strftime('%H:%M:%S')} based on real-time port activity.
    """
    
    send_email_alert(
        recipient=email_config["recipient"],
        subject=subject,
        body=body,
        smtp_server=email_config["smtp_server"],
        smtp_port=email_config["smtp_port"],
        smtp_user=email_config["smtp_user"],
        smtp_password=email_config["smtp_password"]
    )