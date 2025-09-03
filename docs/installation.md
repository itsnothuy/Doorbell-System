# Installation Guide

This guide walks you through setting up the Doorbell Face Recognition Security System on your Raspberry Pi.

## Prerequisites

### Hardware
- Raspberry Pi 4 (2GB+ RAM recommended)
- Raspberry Pi Camera Module v2 or v3
- MicroSD Card (32GB+, Class 10)
- Doorbell button/switch
- Optional: LED indicators (Red, Yellow, Green)
- Weatherproof enclosure
- 5V 3A Power Supply

### Software
- Raspberry Pi OS (Bullseye or newer)
- Internet connection
- Telegram account

## Step 1: Prepare Raspberry Pi OS

### 1.1 Flash Raspberry Pi OS
1. Download [Raspberry Pi Imager](https://www.raspberrypi.org/software/)
2. Flash Raspberry Pi OS (64-bit recommended) to SD card
3. Enable SSH and configure WiFi in imager settings
4. Insert SD card and boot Pi

### 1.2 Initial Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Enable camera interface
sudo raspi-config
# Navigate to: Interface Options → Camera → Enable

# Reboot
sudo reboot
```

## Step 2: Hardware Connections

### 2.1 Camera Connection
1. Turn off Raspberry Pi
2. Connect camera ribbon cable to CSI port
3. Ensure connection is secure

### 2.2 Doorbell Button Wiring
```
Doorbell Button → GPIO 18
Button Ground   → Ground Pin
```

### 2.3 Optional LED Indicators
```
Red LED    → GPIO 16 + 220Ω resistor → Ground
Yellow LED → GPIO 20 + 220Ω resistor → Ground  
Green LED  → GPIO 21 + 220Ω resistor → Ground
```

## Step 3: Software Installation

### 3.1 Clone Repository
```bash
git clone https://github.com/itsnothuy/Doorbell-System.git
cd Doorbell-System
```

### 3.2 Run Automated Setup
```bash
chmod +x setup.sh
./setup.sh
```

The setup script will:
- Install system dependencies
- Create Python virtual environment
- Install Python packages
- Set up project directories
- Configure systemd service
- Create management scripts

## Step 4: Telegram Bot Setup

### 4.1 Create Telegram Bot
1. Message [@BotFather](https://t.me/botfather) on Telegram
2. Send `/newbot` command
3. Follow prompts to create bot
4. Save the bot token

### 4.2 Get Chat ID
1. Message [@userinfobot](https://t.me/userinfobot) on Telegram
2. Note your chat ID

### 4.3 Configure Credentials
```bash
# Edit credentials file
nano config/credentials_telegram.py

# Add your values:
TELEGRAM_BOT_TOKEN = "YOUR_BOT_TOKEN_HERE"
TELEGRAM_CHAT_ID = "YOUR_CHAT_ID_HERE"
```

## Step 5: Face Database Setup

### 5.1 Add Known Faces
```bash
# Add photos of family/friends (one per person)
cp alice.jpg data/known_faces/
cp bob.jpg data/known_faces/
cp charlie.jpg data/known_faces/
```

**Image Requirements:**
- Clear, front-facing photos
- JPEG format
- At least 200x200 pixels
- One face per image
- Good lighting

### 5.2 FBI Blacklist (Optional)
The system automatically downloads FBI most wanted photos. To disable:
```bash
# Edit environment file
nano .env

# Set:
FBI_UPDATE_ENABLED=False
```

## Step 6: Testing

### 6.1 Component Test
```bash
./scripts/test.sh
```

This tests:
- Camera functionality
- Telegram notifications
- GPIO controls
- Face recognition system

### 6.2 Manual Test
```bash
# Activate virtual environment
source venv/bin/activate

# Run system manually
python3 src/doorbell_security.py
```

Press the doorbell button to test detection.

## Step 7: System Service

### 7.1 Start Service
```bash
# Start the service
./scripts/start.sh

# Check status
./scripts/status.sh
```

### 7.2 Enable Auto-Start
```bash
# Service is automatically enabled by setup.sh
# Verify with:
sudo systemctl is-enabled doorbell-security.service
```

## Configuration Options

### Environment Variables
Edit `.env` file to customize:

```bash
# GPIO pins
DOORBELL_PIN=18
RED_LED_PIN=16
YELLOW_LED_PIN=20
GREEN_LED_PIN=21

# Camera settings
CAMERA_RESOLUTION=1280,720
CAMERA_ROTATION=0

# Recognition settings
FACE_TOLERANCE=0.6
BLACKLIST_TOLERANCE=0.5

# Notifications
NOTIFY_ON_KNOWN_VISITORS=False
```

### Advanced Settings
Edit `config/settings.py` for more options.

## Troubleshooting

### Camera Issues
```bash
# Test camera manually
libcamera-still -o test.jpg

# Check camera detection
vcgencmd get_camera
```

### Permission Issues
```bash
# Add user to video group
sudo usermod -a -G video $USER

# Logout and login again
```

### GPIO Issues
```bash
# Add user to gpio group
sudo usermod -a -G gpio $USER

# Check GPIO permissions
ls -l /dev/gpiomem
```

### Service Issues
```bash
# Check service logs
sudo journalctl -u doorbell-security.service -f

# Check application logs
tail -f data/logs/doorbell.log
```

### Memory Issues
```bash
# Increase swap file
sudo dphys-swapfile swapoff
sudo nano /etc/dphys-swapfile
# Set CONF_SWAPSIZE=1024
sudo dphys-swapfile setup
sudo dphys-swapfile swapon
```

## Security Considerations

1. **Network Security**
   - Change default passwords
   - Use strong WiFi passwords
   - Consider VPN for remote access

2. **Physical Security**
   - Secure Pi enclosure
   - Protect camera from tampering
   - Use proper cable management

3. **Data Protection**
   - Regular backups of face database
   - Secure Telegram credentials
   - Monitor access logs

## Performance Optimization

### For Raspberry Pi 3
```bash
# Reduce camera resolution
# Edit .env:
CAMERA_RESOLUTION=640,480

# Reduce face recognition accuracy for speed
FACE_TOLERANCE=0.7
```

### For Better Performance
- Use Raspberry Pi 4 with 4GB+ RAM
- Use fast MicroSD card (Class 10, U3)
- Ensure adequate cooling
- Use quality power supply

## Maintenance

### Regular Tasks
- Check system logs weekly
- Update face database as needed
- Backup configuration monthly
- Update system packages monthly

### Updates
```bash
# Update system
git pull origin main
./setup.sh

# Restart service
./scripts/stop.sh
./scripts/start.sh
```

## Support

For issues and questions:
1. Check logs in `data/logs/`
2. Run diagnostic test: `./scripts/test.sh`
3. Review troubleshooting section
4. Create GitHub issue with logs and system info

---

**Next:** [Configuration Guide](configuration.md)
