# Doorbell Face Recognition Security System

A low-cost, Raspberry Pi-based home security system that recognizes faces at your doorbell and sends alerts via Telegram.

## Features

- **Face Recognition**: Detect and recognize faces using advanced deep learning models
- **Whitelist Management**: Maintain a database of known/authorized persons
- **Blacklist Integration**: Cross-check against FBI most wanted list
- **Real-time Alerts**: Instant Telegram notifications with photos
- **Local Processing**: All face recognition happens locally for privacy
- **Low Cost**: Built with Raspberry Pi 4 and common components (~$100 total)

## System Overview

When someone presses the doorbell:
1. Camera captures visitor's image
2. System detects and analyzes faces
3. Compares against known faces (whitelist) and suspicious faces (blacklist)
4. Sends appropriate alert via Telegram with classification:
   - 🟢 **Known Person**: Recognized family/friend (logged, no alert)
   - 🟡 **Unknown Person**: Unrecognized visitor (photo alert sent)
   - 🔴 **Blacklisted Person**: Potential security threat (urgent alert)

## Architecture

This system follows a **modular monolith** architecture inspired by [Frigate NVR](https://frigate.video/). For detailed architectural decisions, see our [Architecture Decision Records (ADRs)](docs/adr/README.md):

- [ADR-001: System Architecture](docs/adr/ADR-001-system-architecture.md) - Modular monolith design
- [ADR-002: Face Recognition Implementation](docs/adr/ADR-002-face-recognition-implementation.md) - Two-stage detection/recognition pipeline  
- [ADR-003: Generative AI Integration](docs/adr/ADR-003-generative-ai-integration.md) - Optional AI descriptions
- [ADR-004: Testing Strategy](docs/adr/ADR-004-testing-strategy.md) - Layered testing approach

## Hardware Requirements

### 🥧 Production Deployment (Raspberry Pi)
- Raspberry Pi 4 (4GB RAM recommended)
- Raspberry Pi Camera Module v2 or v3
- MicroSD Card (32GB+)
- Doorbell button/switch
- LED indicators (optional)
- Weatherproof enclosure
- 5V 3A Power Supply

**Total Cost: ~$100**

### 🍎 Development/Testing (macOS/PC)
- MacBook Air/Pro (M1/M2/Intel) or any PC
- Built-in webcam or USB camera
- Internet connection
- Telegram account (for notifications)

**Total Cost: $0** (uses existing hardware)

## Quick Start

Choose your deployment option:

<details>
<summary>🍎 <strong>macOS Development/Testing (Recommended for Testing)</strong></summary>

Perfect for testing all functionality using your MacBook's webcam!

### 1. Clone & Setup
```bash
git clone https://github.com/itsnothuy/Doorbell-System.git
cd Doorbell-System

# Run macOS setup
chmod +x setup-macos.sh
./setup-macos.sh
```

### 2. Configure (Optional)
```bash
# Set up Telegram bot credentials
nano config/credentials_telegram.py
# Add your bot token and chat ID

# Add known faces (optional)
# Add photos: data/known_faces/alice.jpg, etc.
```

### 3. Start & Test
```bash
# Test all components
./scripts/test-macos.sh

# Start web interface
./scripts/start-web.sh

# Open browser to http://localhost:5000
```

**✨ Features on macOS:**
- 📷 Uses built-in webcam
- 🌐 Beautiful web interface for testing
- 🔘 Virtual doorbell button (click to trigger)
- 📱 Real Telegram notifications
- 👥 Full face recognition testing
- 💡 Virtual LED status indicators

</details>

<details>
<summary>🥧 <strong>Raspberry Pi Production</strong></summary>

### 1. Hardware Setup
```bash
# Connect camera to Pi's CSI port
# Wire doorbell button to GPIO 18 and Ground
# Mount camera at door with clear view of visitor's face height
```

### 2. Software Installation
```bash
# Clone this repository
git clone https://github.com/itsnothuy/Doorbell-System.git
cd Doorbell-System

# Run automated setup
chmod +x setup.sh
./setup.sh
```

### 3. Configuration
```bash
# Set up Telegram bot credentials
cp config/credentials_template.py config/credentials_telegram.py
# Edit with your bot token and chat ID

# Add known faces
mkdir -p data/known_faces
# Add photos: alice.jpg, bob.jpg, etc.
```

### 4. Start System
```bash
# Install as system service
sudo ./install_service.sh

# Or run manually for testing
python3 src/doorbell_security.py
```

</details>

<details>
<summary>☁️ <strong>Cloud Deployment (Free)</strong></summary>

Deploy for free testing on cloud platforms:

### Vercel (Recommended)
```bash
# Install Vercel CLI
npm i -g vercel

# Deploy
vercel --prod
```

### Render
1. Connect your GitHub repo to [Render](https://render.com)
2. Uses `render.yaml` for auto-configuration
3. Free tier available

### Railway
```bash
# Install Railway CLI
npm i -g @railway/cli

# Deploy
railway up
```

### Docker (Local)
```bash
# Build and run
docker-compose up

# Access at http://localhost:5000
```

</details>

## 🌐 Web Interface (macOS/Development)

When running on macOS or in development mode, the system includes a beautiful web interface:

![Web Interface Preview](https://via.placeholder.com/800x600/667eea/ffffff?text=Doorbell+Security+Dashboard)

### Features:
- 📊 **Real-time System Status** - Monitor all components
- 🔔 **Virtual Doorbell** - Click to simulate button press  
- 📷 **Live Camera Feed** - Test webcam capture
- 👥 **Face Recognition Test** - See detection results
- 💡 **LED Control** - Test status indicators
- 📱 **Notification Test** - Send test alerts
- 📋 **Event History** - View recent activities
- 📝 **System Logs** - Debug and monitor

### Access:
- Local: `http://localhost:5000`
- Vercel: `https://your-app.vercel.app`
- Render: `https://your-app.onrender.com`

## Architecture

- **Cross-Platform**: Runs on Raspberry Pi, macOS, Linux, Windows
- **Edge Processing**: All face recognition runs locally (Raspberry Pi mode)
- **Web Interface**: Modern dashboard for testing and monitoring
- **Modular Design**: Separate modules for camera, recognition, alerts, etc.
- **Privacy First**: No face data sent to cloud services
- **Fail-Safe**: System defaults to unknown alerts rather than false recognition

## Project Structure

```
Doorbell-System/
├── 📁 src/                     # Source code
│   ├── 🎯 doorbell_security.py    # Main application
│   ├── 👤 face_manager.py         # Face database management
│   ├── 📷 camera_handler.py       # Camera operations  
│   ├── 📱 telegram_notifier.py    # Notification system
│   ├── 🔌 gpio_handler.py         # Hardware interface
│   ├── 🔍 platform_detector.py   # Cross-platform detection
│   └── 🌐 web_interface.py       # Web dashboard
├── 📁 config/                  # Configuration files
│   ├── ⚙️ settings.py             # System settings
│   └── 🔑 credentials_template.py # Telegram credentials template
├── 📁 data/                    # Face databases and logs
│   ├── 👥 known_faces/            # Authorized persons
│   ├── 🚫 blacklist_faces/        # Suspicious persons
│   ├── 📸 captures/               # Visitor snapshots
│   └── 📝 logs/                   # System logs
├── 📁 templates/               # Web interface templates
│   └── 🎨 dashboard.html          # Main dashboard
├── 📁 scripts/                 # Utility scripts
├── 📁 tests/                   # Test suite
│   └── 🧪 test_system.py         # Unit tests
├── 📁 docs/                    # Documentation
├── 🛠️ setup.sh                   # Raspberry Pi setup
├── 🍎 setup-macos.sh             # macOS setup
├── 🌐 app.py                     # Cloud deployment entry
├── 🐳 Dockerfile                 # Docker configuration
├── ☁️ vercel.json               # Vercel deployment
├── 🚄 railway.toml              # Railway deployment
└── 📋 render.yaml               # Render deployment
```

## Documentation

- [Installation Guide](docs/installation.md)
- [Configuration Guide](docs/configuration.md)
- [Hardware Setup](docs/hardware_setup.md)
- [Troubleshooting](docs/troubleshooting.md)
- [API Reference](docs/api_reference.md)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

MIT License - see LICENSE file for details

## Safety & Legal Notice

- This system is for personal home security use only
- Inform visitors that recording is in use (legal requirement in many areas)
- False positives are possible - always verify alerts manually
- Not a replacement for professional security systems
- Use responsibly and in compliance with local privacy laws

## References

Based on comprehensive research into face recognition systems including:
- Google Photos face recognition architecture
- Low-cost development methodologies  
- Open-source computer vision libraries
- Privacy-preserving edge computing approaches

## 🚀 Getting Started on MacBook M4 Air

Perfect for testing the complete system on your MacBook!

### Quick Start (5 minutes):

```bash
# 1. Clone the repository
git clone https://github.com/itsnothuy/Doorbell-System.git
cd Doorbell-System

# 2. Run automated setup
./setup-macos.sh

# 3. Test all components
./scripts/test-macos.sh

# 4. Start web interface  
./scripts/start-web.sh
```

### Then open: http://localhost:5000

🎉 **You now have a fully functional doorbell security system running on your MacBook!**

### What You Can Test:
- ✅ **Face Recognition** using your webcam
- ✅ **Virtual Doorbell** button in web interface  
- ✅ **Telegram Notifications** (if configured)
- ✅ **LED Status Indicators** (virtual)
- ✅ **Event Logging** and history
- ✅ **All System Components** without any hardware

### Deploy to Cloud (Free):
```bash
# Vercel (instant)
npm i -g vercel && vercel --prod

# Railway  
npm i -g @railway/cli && railway up

# Or use GitHub integration with Render/Railway
```

## 💰 Cost Comparison

| Platform | Initial Cost | Monthly Cost | Features |
|----------|-------------|--------------|----------|
| **MacBook Testing** | $0 | $0 | Full testing, web interface |
| **Cloud Deployment** | $0 | $0 | Online demo, sharing |
| **Raspberry Pi** | ~$100 | $0 | Production hardware |
| **Commercial Systems** | $200-500 | $10-30 | Limited features, subscriptions |

**Winner: This system! 🏆**

---

**Built with ❤️ for home security and privacy**

*Perfect for testing on MacBook M4 Air and deploying cost-free to the cloud! 🚀*
