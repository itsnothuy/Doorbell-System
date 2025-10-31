# User Onboarding Guide

## Welcome to Doorbell Security System!

This guide will help you get started with your new AI-powered home security system in just a few minutes.

---

## Table of Contents

1. [Before You Begin](#before-you-begin)
2. [Quick Start (5 Minutes)](#quick-start-5-minutes)
3. [Initial Setup](#initial-setup)
4. [Adding Known Faces](#adding-known-faces)
5. [Configuring Notifications](#configuring-notifications)
6. [Testing Your System](#testing-your-system)
7. [Daily Usage](#daily-usage)
8. [Troubleshooting](#troubleshooting)
9. [Getting Help](#getting-help)

---

## Before You Begin

### What You'll Need

**Hardware** (for Raspberry Pi deployment):
- ✅ Raspberry Pi 4 (4GB+ RAM)
- ✅ Pi Camera Module v2 or USB camera
- ✅ MicroSD card (32GB+, Class 10)
- ✅ Power supply (5V 3A)
- ✅ Internet connection

**Or** (for testing/development):
- ✅ MacBook/PC with webcam
- ✅ Internet connection

**Skills Required**:
- ⭐ Basic computer skills
- ⭐ Ability to follow instructions
- ⭐⭐ Command line basics (optional)

**Time Required**:
- 🕐 5 minutes for testing mode
- 🕐 15-20 minutes for full production setup

---

## Quick Start (5 Minutes)

### Step 1: Download the Software

```bash
# Clone the repository
git clone https://github.com/itsnothuy/Doorbell-System.git
cd Doorbell-System
```

### Step 2: Run Setup Script

**On macOS/Linux**:
```bash
chmod +x setup-macos.sh
./setup-macos.sh
```

**On Raspberry Pi**:
```bash
chmod +x setup.sh
./setup.sh
```

### Step 3: Start the System

```bash
# Start web interface
python3 app.py
```

### Step 4: Access Dashboard

Open your browser and go to:
```
http://localhost:5000
```

🎉 **That's it!** You're now running the Doorbell Security System!

---

## Initial Setup

### Configuration Wizard

When you first start the system, you'll be guided through a setup wizard:

#### 1. System Settings (30 seconds)

- **Device Name**: Give your doorbell a name (e.g., "Front Door")
- **Timezone**: Select your timezone
- **Language**: Choose your preferred language
- **Admin Email**: Enter your email for alerts

#### 2. Camera Configuration (45 seconds)

- **Camera Source**: System will auto-detect your camera
- **Resolution**: Recommended 640x480 for best performance
- **Frame Rate**: 15 FPS (default)
- **Detection Zones**: Draw areas where faces should be detected

**Tips**:
- Position camera at door height (5-6 feet)
- Ensure good lighting
- Test with the "Preview" button

#### 3. Face Recognition Settings (30 seconds)

- **Recognition Threshold**: 0.6 (default, recommended)
  - Lower = more strict (fewer false positives)
  - Higher = more lenient (catch more faces)
- **Unknown Alert Frequency**: How often to alert for unknown persons
- **Face Model**: "Large (accurate)" recommended

#### 4. Notification Setup (60 seconds)

Choose how you want to be notified:

- ✅ **Web Dashboard**: Always enabled
- ☐ **Email Alerts**: Requires SMTP setup
- ☐ **Telegram**: Requires bot token (optional)
- ☐ **Push Notifications**: Mobile app (coming soon)

**Email Setup** (if enabled):
```
SMTP Server: smtp.gmail.com
SMTP Port: 587
Username: your-email@gmail.com
Password: your-app-password
```

💡 **Tip**: For Gmail, use an [App Password](https://support.google.com/accounts/answer/185833)

#### 5. Security Settings (45 seconds)

- **Enable Authentication**: Recommended for web access
- **Session Timeout**: 30 minutes (default)
- **Max Login Attempts**: 3 before lockout
- **Two-Factor Auth**: Optional additional security

---

## Adding Known Faces

### Why Add Known Faces?

- 🏠 Family members won't trigger alerts
- 👥 Track who visits your home
- 📊 Better analytics and insights

### How to Add a Person

#### Method 1: Web Dashboard (Easiest)

1. Go to **Dashboard** → **Known Faces**
2. Click **"Add Person"**
3. Enter name (e.g., "John Smith")
4. Upload 2-3 photos:
   - Front-facing photo
   - Left profile (optional)
   - Right profile (optional)
5. Click **"Register"**

#### Method 2: File System (Advanced)

```bash
# Add photos to known_faces directory
cp photos/john.jpg data/known_faces/john_smith.jpg
cp photos/sarah.jpg data/known_faces/sarah_smith.jpg

# Restart system to load new faces
sudo systemctl restart doorbell
```

### Best Practices for Photos

✅ **Good Photos**:
- Clear, well-lit face
- Looking at camera
- Neutral expression
- No sunglasses or masks
- High resolution (640x480 minimum)

❌ **Avoid**:
- Blurry or dark photos
- Side profiles only
- Hats or hoods covering face
- Multiple people in photo
- Low resolution images

### Testing Recognition

After adding faces:

1. Go to **Dashboard** → **Test Recognition**
2. Upload a test photo or use webcam
3. System will identify the person
4. Check confidence score (should be >0.8)

If recognition fails:
- Add more photos from different angles
- Ensure good lighting
- Check photo quality

---

## Configuring Notifications

### Notification Types

**Known Person Detected**:
- Silent notification (no alert)
- Logged for history
- Optional "Welcome home" message

**Unknown Person Detected**:
- Immediate alert
- Photo attached
- Timestamp and location

**System Events**:
- Camera offline
- Low storage
- System errors

### Notification Channels

#### Web Dashboard
Always active, no configuration needed.

Access: `http://your-doorbell-ip:5000`

#### Email Alerts

1. Go to **Settings** → **Notifications**
2. Enable **"Email Alerts"**
3. Configure SMTP settings
4. Test with **"Send Test Email"**

#### Telegram (Optional)

1. Create a Telegram bot:
   - Message @BotFather on Telegram
   - Use `/newbot` command
   - Save your bot token

2. Get your chat ID:
   - Message your bot
   - Visit: `https://api.telegram.org/bot<TOKEN>/getUpdates`
   - Find your chat ID in the response

3. Configure in system:
   ```python
   # config/credentials_telegram.py
   TELEGRAM_BOT_TOKEN = "your-bot-token"
   TELEGRAM_CHAT_ID = "your-chat-id"
   ```

### Notification Schedules

Customize when you receive alerts:

- **24/7**: Always receive alerts
- **Working Hours**: 9am-5pm only
- **Night Mode**: 10pm-6am only
- **Custom**: Define your own schedule

---

## Testing Your System

### Pre-Flight Check

Run the built-in test suite:

```bash
# Quick system check
python3 -m demo.flows.troubleshooting

# Run all tests
python3 -m demo.orchestrator --quick
```

### Manual Testing

#### Test 1: Camera Feed

1. Open dashboard
2. Check live camera feed
3. Verify image quality
4. Test day/night vision

#### Test 2: Face Detection

1. Stand in front of camera
2. Check if face box appears
3. Verify detection confidence
4. Test different angles

#### Test 3: Face Recognition

1. Register your face
2. Stand in front of camera
3. Verify you're recognized
4. Check confidence score

#### Test 4: Notifications

1. Configure notification channel
2. Send test notification
3. Verify receipt
4. Check notification content

### Performance Benchmarks

Your system should achieve:
- Face Detection: < 0.2s
- Face Recognition: < 0.5s
- Total Processing: < 1.0s
- Accuracy: > 95%

If performance is slow:
- Reduce camera resolution
- Lower frame rate
- Enable hardware acceleration
- Check CPU/memory usage

---

## Daily Usage

### Normal Operation

Once configured, the system runs automatically:

1. **Motion Detected** → Camera activates
2. **Face Detected** → Recognition begins
3. **Person Identified** → Log and notify
4. **Event Stored** → History updated

### Viewing Events

**Dashboard Timeline**:
- Shows recent activity
- Click event for details
- Filter by person/time
- Export event data

**Event Details**:
- Timestamp
- Person identified
- Confidence score
- Photo/video
- Actions taken

### Managing Known Faces

**Update Person**:
1. Go to **Known Faces**
2. Click person name
3. Add/remove photos
4. Update details

**Remove Person**:
1. Select person
2. Click **"Remove"**
3. Confirm deletion
4. Face data is deleted

### System Maintenance

**Weekly Tasks**:
- Review event log
- Check storage usage
- Update known faces
- Test notifications

**Monthly Tasks**:
- System backup
- Performance review
- Update software
- Clean camera lens

---

## Troubleshooting

### Common Issues

#### Camera Not Working

**Symptoms**: No video feed, camera offline

**Solutions**:
1. Check camera connection
2. Verify camera permissions:
   ```bash
   sudo raspi-config
   # Enable camera under Interface Options
   ```
3. Test camera:
   ```bash
   raspistill -o test.jpg
   ```
4. Restart camera service:
   ```bash
   sudo systemctl restart doorbell-camera
   ```

#### Face Not Recognized

**Symptoms**: Known person shows as unknown

**Solutions**:
1. Add more photos (different angles)
2. Check photo quality
3. Adjust recognition threshold
4. Improve lighting
5. Rebuild face database:
   ```bash
   python3 -m src.face_manager --rebuild
   ```

#### Slow Performance

**Symptoms**: Long processing times, delayed notifications

**Solutions**:
1. Check CPU usage:
   ```bash
   top
   ```
2. Reduce camera resolution
3. Lower frame rate
4. Free up memory:
   ```bash
   sudo systemctl restart doorbell
   ```
5. Enable hardware acceleration

#### Notifications Not Working

**Symptoms**: No alerts received

**Solutions**:
1. Check notification settings
2. Test notification channel
3. Verify internet connection
4. Check spam folder (email)
5. Verify credentials (Telegram/SMTP)

### Getting Diagnostics

Run automated diagnostics:

```bash
python3 -m demo.flows.troubleshooting
```

This checks:
- Camera connectivity
- Face detection engine
- Database access
- Network connection
- Storage space
- System temperature

---

## Getting Help

### Documentation

- **Full Documentation**: `docs/`
- **API Reference**: `docs/API.md`
- **Architecture**: `docs/ARCHITECTURE.md`
- **FAQ**: `docs/FAQ.md`

### Community Support

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share ideas
- **Discord**: Real-time community chat
- **Reddit**: r/HomeAutomation

### Professional Support

For commercial deployments or custom integrations:
- **Email**: support@doorbell-system.example.com
- **Consultation**: Schedule a call
- **Custom Development**: Request a quote

---

## Next Steps

### Explore Advanced Features

Once you're comfortable with basics:

1. **Multi-Camera Setup**: Add more cameras
2. **Pattern Recognition**: Enable AI analysis
3. **Automation**: Integrate with home automation
4. **API Integration**: Connect to other services
5. **Custom Notifications**: Build custom handlers

### Join the Community

- ⭐ Star the project on GitHub
- 📝 Share your setup
- 🐛 Report issues
- 💡 Suggest features
- 🤝 Contribute code

### Stay Updated

- Follow on Twitter: @DoorbellSecurity
- Subscribe to newsletter
- Check releases page
- Read the blog

---

## Congratulations!

You've successfully set up your Doorbell Security System! 🎉

Your home is now protected by:
- ✅ Real-time face recognition
- ✅ Intelligent notifications
- ✅ Comprehensive event logging
- ✅ Privacy-first architecture

Enjoy your new security system!

---

**Need Help?** Join our community or check the documentation.

**Happy?** Give us a star on GitHub! ⭐

**Version**: 1.0.0
**Last Updated**: October 31, 2024
