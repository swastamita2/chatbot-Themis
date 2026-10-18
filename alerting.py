"""
alerting.py - Sistem Notifikasi Alert (Clean Architecture & OOP)
Menyediakan antarmuka terpusat (AlertManager) dengan berbagai implementasi notifier:

1. REST API Notifiers (HTTPS Port 443 - BEBAS BLOKIR ISP / HOTSPOT TANPA VPN):
   - ResendEmailNotifier: Kirim email via Resend REST API (resend.com, 100 email/hari gratis)
   - GoogleScriptEmailNotifier: Kirim email via Google Apps Script Webhook (gratis dari Gmail asli)
   - BrevoEmailNotifier: Kirim email via Brevo REST API (300 email/hari gratis)
   - TelegramNotifier: Kirim alert via Telegram Bot API
   - DiscordNotifier: Kirim alert via Discord Webhook

2. Legacy SMTP Notifiers (Port 587/465 - Rentan diblokir ISP/Seluler):
   - SmtpEmailNotifier: Kirim via SMTP langsung

3. Fallback:
   - ConsoleNotifier: Output teks ke console/terminal
   - UnifiedEmailNotifier: Secara otomatis mendeteksi provider HTTPS aktif,
     dan hanya fallback ke SMTP jika provider REST API belum dikonfigurasi.
"""

import os
import smtplib
from abc import ABC, abstractmethod
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

# Muat environment variables
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")


class BaseNotifier(ABC):
    """Antarmuka dasar (Interface) untuk seluruh penyedia notifikasi alert."""

    @abstractmethod
    def is_configured(self) -> bool:
        """Cek apakah konfigurasi/kredensial penyedia sudah lengkap."""
        pass

    @abstractmethod
    def send(self, subject: str, message: str, context: Optional[Dict[str, Any]] = None) -> bool:
        """Kirim notifikasi. Mengembalikan True jika berhasil, False jika gagal."""
        pass


class ConsoleNotifier(BaseNotifier):
    """Fallback notifier ke console/terminal dengan format teks aman."""

    def is_configured(self) -> bool:
        return True

    def send(self, subject: str, message: str, context: Optional[Dict[str, Any]] = None) -> bool:
        print("\n" + "=" * 60)
        print(f"[ALERT TRIGGERED] Topik: {subject}")
        print(f"Detail:\n{message}")
        if context:
            print(f"Context: {context}")
        print("=" * 60 + "\n")
        return True


class TelegramNotifier(BaseNotifier):
    """Implementasi notifikasi via Telegram Bot API (HTTPS Port 443 - Anti Blokir)."""

    def __init__(self, token: Optional[str] = None, chat_id: Optional[str] = None):
        self.token = token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")

    def is_configured(self) -> bool:
        return bool(self.token and self.chat_id and "xxxx" not in self.token)

    def send(self, subject: str, message: str, context: Optional[Dict[str, Any]] = None) -> bool:
        if not self.is_configured():
            return False

        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": f"<b>[{subject}]</b>\n\n{message}",
            "parse_mode": "HTML",
        }
        try:
            resp = requests.post(url, json=payload, timeout=6)
            if resp.status_code == 200:
                print("[ALERT - TELEGRAM] Notifikasi berhasil terkirim ke Telegram.")
                return True
            else:
                print(f"[ALERT - TELEGRAM] Gagal (HTTP {resp.status_code}): {resp.text}")
                return False
        except Exception as e:
            print(f"[ALERT - TELEGRAM] Gagal terhubung ke Telegram API: {e}")
            return False


class DiscordNotifier(BaseNotifier):
    """Implementasi notifikasi via Discord Webhook (HTTPS Port 443 - Anti Blokir)."""

    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("DISCORD_WEBHOOK_URL")

    def is_configured(self) -> bool:
        return bool(self.webhook_url and self.webhook_url.startswith("https://discord.com/api/webhooks"))

    def send(self, subject: str, message: str, context: Optional[Dict[str, Any]] = None) -> bool:
        if not self.is_configured():
            return False

        payload = {
            "embeds": [
                {
                    "title": f"🚨 [AI-ALERT] {subject}",
                    "description": message[:2000],
                    "color": 0xFF4D4D,
                    "timestamp": datetime.now().astimezone().isoformat(),
                }
            ]
        }
        try:
            resp = requests.post(self.webhook_url, json=payload, timeout=6)
            if resp.status_code in (200, 204):
                print("[ALERT - DISCORD] Notifikasi berhasil terkirim ke Discord channel.")
                return True
            else:
                print(f"[ALERT - DISCORD] Gagal (HTTP {resp.status_code}): {resp.text}")
                return False
        except Exception as e:
            print(f"[ALERT - DISCORD] Gagal terhubung ke Discord Webhook: {e}")
            return False


class ResendEmailNotifier(BaseNotifier):
    """
    Kirim Email via Resend REST API (HTTPS Port 443).
    Sangat direkomendasikan untuk developer (gratis 3.000 email/bulan, tanpa verifikasi kartu kredit).
    Daftar di https://resend.com, dapatkan API Key (format re_xxx).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        receiver: Optional[str] = None,
        from_email: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("RESEND_API_KEY")
        self.receiver = receiver or os.getenv("EMAIL_RECEIVER") or os.getenv("EMAIL_SENDER")
        # onboarding@resend.dev adalah sender sandbox resmi yang bisa langsung kirim ke email pribadi akun
        self.from_email = from_email or os.getenv("RESEND_FROM", "AI Alert <onboarding@resend.dev>")

    def is_configured(self) -> bool:
        return bool(self.api_key and self.receiver and self.api_key.startswith("re_"))

    def send(self, subject: str, message: str, context: Optional[Dict[str, Any]] = None) -> bool:
        if not self.is_configured():
            return False

        url = "https://api.resend.com/emails"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        # Format baris baru menjadi HTML sederhana untuk email yang rapi
        html_content = message.replace("\n", "<br>")
        payload = {
            "from": self.from_email,
            "to": [self.receiver],
            "subject": f"[AI-ALERT] {subject}",
            "text": message,
            "html": f"<h3>[AI-ALERT] {subject}</h3><p>{html_content}</p>",
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=8)
            if resp.status_code in (200, 201):
                data = resp.json()
                print(f"[ALERT - RESEND] Email berhasil dikirim ke {self.receiver} via HTTPS (ID: {data.get('id')}).")
                return True
            else:
                print(f"[ALERT - RESEND] Gagal (HTTP {resp.status_code}): {resp.text}")
                return False
        except Exception as e:
            print(f"[ALERT - RESEND] Error koneksi HTTPS: {e}")
            return False


class GoogleScriptEmailNotifier(BaseNotifier):
    """
    Kirim Email via Google Apps Script Webhook (HTTPS Port 443).
    Email dikirim langsung dari akun Gmail pengguna sendiri tanpa API pihak ketiga!
    Hanya perlu 1 Webhook URL dari script.google.com (gratis, limit 500 email/hari).
    """

    def __init__(self, webhook_url: Optional[str] = None, receiver: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("GMAIL_WEBHOOK_URL")
        self.receiver = receiver or os.getenv("EMAIL_RECEIVER") or os.getenv("EMAIL_SENDER")

    def is_configured(self) -> bool:
        return bool(self.webhook_url and self.webhook_url.startswith("https://script.google.com"))

    def send(self, subject: str, message: str, context: Optional[Dict[str, Any]] = None) -> bool:
        if not self.is_configured():
            return False

        payload = {
            "to": self.receiver,
            "subject": f"[AI-ALERT] {subject}",
            "body": message,
        }

        try:
            # Google Apps Script web app melakukan redirect 302 ke output
            resp = requests.post(self.webhook_url, json=payload, timeout=10, allow_redirects=True)
            if resp.status_code in (200, 302):
                print(f"[ALERT - GMAIL SCRIPT] Email berhasil dikirim ke {self.receiver} via Google Script HTTPS.")
                return True
            else:
                print(f"[ALERT - GMAIL SCRIPT] Gagal (HTTP {resp.status_code}): {resp.text}")
                return False
        except Exception as e:
            print(f"[ALERT - GMAIL SCRIPT] Error koneksi ke Google Apps Script: {e}")
            return False


class BrevoEmailNotifier(BaseNotifier):
    """Kirim Email via Brevo (Sendinblue) REST API v3 (HTTPS Port 443)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        receiver: Optional[str] = None,
        sender_email: Optional[str] = None,
    ):
        self.api_key = api_key or os.getenv("BREVO_API_KEY")
        self.receiver = receiver or os.getenv("EMAIL_RECEIVER") or os.getenv("EMAIL_SENDER")
        self.sender_email = sender_email or os.getenv("BREVO_SENDER_EMAIL") or os.getenv("EMAIL_SENDER")

    def is_configured(self) -> bool:
        return bool(self.api_key and self.receiver and self.sender_email and "xxxx" not in self.api_key)

    def send(self, subject: str, message: str, context: Optional[Dict[str, Any]] = None) -> bool:
        if not self.is_configured():
            return False

        url = "https://api.brevo.com/v3/smtp/email"
        headers = {
            "api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        payload = {
            "sender": {"name": "AI Agent Alert", "email": self.sender_email},
            "to": [{"email": self.receiver}],
            "subject": f"[AI-ALERT] {subject}",
            "textContent": message,
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=8)
            if resp.status_code in (200, 201):
                print(f"[ALERT - BREVO] Email berhasil dikirim ke {self.receiver} via Brevo REST API.")
                return True
            else:
                print(f"[ALERT - BREVO] Gagal (HTTP {resp.status_code}): {resp.text}")
                return False
        except Exception as e:
            print(f"[ALERT - BREVO] Error koneksi ke Brevo: {e}")
            return False


class SmtpEmailNotifier(BaseNotifier):
    """
    Implementasi notifikasi email via SMTP langsung.
    CATATAN: Port SMTP (587/465) sering diblokir oleh ISP/Hotspot di Indonesia.
    """

    def __init__(
        self,
        sender: Optional[str] = None,
        app_password: Optional[str] = None,
        receiver: Optional[str] = None,
        smtp_server: Optional[str] = None,
        smtp_port: Optional[int] = None,
    ):
        self.sender = sender or os.getenv("EMAIL_SENDER")
        raw_pwd = app_password or os.getenv("EMAIL_APP_PASSWORD", "")
        self.app_password = raw_pwd.replace(" ", "")
        self.receiver = receiver or os.getenv("EMAIL_RECEIVER")
        self.smtp_server = smtp_server or os.getenv("SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = smtp_port or int(os.getenv("SMTP_PORT", "587"))

    def is_configured(self) -> bool:
        return bool(
            self.sender
            and self.app_password
            and self.receiver
            and "xxxxxxxx" not in self.app_password
        )

    def _print_blocked_diagnosis(self, error_tls: Exception, error_ssl: Exception) -> None:
        """Cetak diagnosis port SMTP diblokir + solusi REST API HTTPS."""
        print(f"[ALERT - EMAIL SMTP] Gagal koneksi SMTP (TLS: {error_tls} | SSL: {error_ssl}).", flush=True)
        print("  [DIAGNOSIS] Port SMTP (587 & 465) diblokir oleh ISP/Hotspot seluler Anda.", flush=True)
        print("  [SOLUSI] Gunakan REST API via HTTPS (Port 443) yang 100% bebas blokir:", flush=True)
        print("           1. Resend API (set RESEND_API_KEY di .env) -> Sangat cepat & gratis", flush=True)
        print("           2. Google Apps Script Webhook (set GMAIL_WEBHOOK_URL di .env)", flush=True)
        print("           3. Brevo REST API (set BREVO_API_KEY di .env)", flush=True)

    def send(self, subject: str, message: str, context: Optional[Dict[str, Any]] = None) -> bool:
        if not self.is_configured():
            return False

        msg = MIMEMultipart()
        msg["From"] = self.sender
        msg["To"] = self.receiver
        msg["Subject"] = f"[AI-ALERT] {subject}"
        msg.attach(MIMEText(message, "plain"))

        import socket
        old_timeout = socket.getdefaulttimeout()
        socket.setdefaulttimeout(2.5)
        try:
            # Percobaan 1: SMTP TLS (Port 587)
            try:
                with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=2.5) as server:
                    server.starttls()
                    server.login(self.sender, self.app_password)
                    server.send_message(msg)
                print(f"[ALERT - EMAIL SMTP] Email terkirim ke {self.receiver} (via Port {self.smtp_port}).", flush=True)
                return True
            except Exception as e_tls:
                # Percobaan 2: Fallback ke SMTP SSL (Port 465)
                try:
                    with smtplib.SMTP_SSL(self.smtp_server, 465, timeout=2.5) as server_ssl:
                        server_ssl.login(self.sender, self.app_password)
                        server_ssl.send_message(msg)
                    print(f"[ALERT - EMAIL SMTP] Email terkirim ke {self.receiver} (via Port 465 SSL).", flush=True)
                    return True
                except Exception as e_ssl:
                    self._print_blocked_diagnosis(e_tls, e_ssl)
                    return False
        finally:
            socket.setdefaulttimeout(old_timeout)


class UnifiedEmailNotifier(BaseNotifier):
    """
    Email Notifier cerdas yang memprioritaskan REST API HTTPS (Port 443)
    agar terhindar dari pemblokiran port SMTP oleh ISP/Hotspot.
    """

    def __init__(self):
        self.resend = ResendEmailNotifier()
        self.google_script = GoogleScriptEmailNotifier()
        self.brevo = BrevoEmailNotifier()
        self.smtp = SmtpEmailNotifier()

    def _providers(self):
        """Provider email berurutan sesuai prioritas:
        REST API HTTPS (Port 443, bebas blokir ISP) dulu, SMTP terakhir."""
        return (self.resend, self.google_script, self.brevo, self.smtp)

    def is_configured(self) -> bool:
        return any(p.is_configured() for p in self._providers())

    def send(self, subject: str, message: str, context: Optional[Dict[str, Any]] = None) -> bool:
        for provider in self._providers():
            if provider.is_configured():
                return provider.send(subject, message, context)
        print("[ALERT - EMAIL] Belum ada konfigurasi email yang valid di .env.")
        return False


class AlertManager:
    """Pengelola terpusat untuk memicu alert ke seluruh notifier terdaftar."""

    def __init__(self, notifiers: Optional[List[BaseNotifier]] = None):
        self.notifiers: List[BaseNotifier] = notifiers or [
            ConsoleNotifier(),
            UnifiedEmailNotifier(),
            TelegramNotifier(),
            DiscordNotifier(),
        ]

    def add_notifier(self, notifier: BaseNotifier):
        self.notifiers.append(notifier)

    def trigger_alert(
        self,
        judul: str,
        detail: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Picu pengiriman alert ke seluruh saluran yang terpasang."""
        waktu = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        body = f"Waktu: {waktu}\nTopik: {judul}\nDetail: {detail}\n"
        if context:
            body += "Context:\n"
            for k, v in context.items():
                body += f"- {k}: {v}\n"

        results = {}
        for notifier in self.notifiers:
            name = notifier.__class__.__name__
            try:
                results[name] = notifier.send(judul, body, context)
            except Exception as e:
                results[name] = False
                print(f"[WARN] Error pada {name}: {e}")

        return {
            "status": "alert_triggered",
            "timestamp": waktu,
            "results": results,
        }

    def send_alert(
        self,
        subject: str,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Alias kompatibilitas untuk trigger_alert."""
        return self.trigger_alert(judul=subject, detail=message, context=context)


# Singleton manager default
alert_manager = AlertManager()


if __name__ == "__main__":
    print("=== Menguji Sistem Alerting Multi-Channel (Clean OOP) ===")
    alert_manager.trigger_alert(
        judul="Uji Coba Alert Multi-Provider",
        detail="Verifikasi AlertManager mendukung REST API HTTPS & SMTP fallback.",
        context={"environment": "benchmark-agent-mcp", "test_id": 102},
    )
