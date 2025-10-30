/**
 * Doorbell Security System - Enhanced Dashboard
 * 
 * Enhanced interactivity and real-time features for the dashboard
 */

class Dashboard {
  constructor() {
    this.updateInterval = null;
    this.eventStreamClient = null;
    this.webSocketClient = null;
    this.notificationSystem = new NotificationSystem();
    
    // State management
    this.state = {
      systemOnline: false,
      lastUpdate: null,
      sidebarOpen: false
    };
    
    this.init();
  }
  
  init() {
    console.log('Initializing enhanced dashboard...');
    
    // Setup event listeners
    this.setupEventListeners();
    
    // Setup mobile navigation
    this.setupMobileNav();
    
    // Initialize streaming
    this.initializeStreaming();
    
    // Initial data load
    this.refreshAll();
    
    // Auto-refresh every 30 seconds
    this.updateInterval = setInterval(() => {
      this.refreshStatus();
      this.refreshEvents();
    }, 30000);
    
    // Request notification permission
    this.requestNotificationPermission();
  }
  
  setupEventListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refresh-data');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', () => this.refreshAll());
    }
    
    // Navigation items - add active class on click
    document.querySelectorAll('.nav-item').forEach(item => {
      item.addEventListener('click', (e) => {
        document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
        e.currentTarget.classList.add('active');
      });
    });
  }
  
  setupMobileNav() {
    const toggleBtn = document.getElementById('mobile-nav-toggle');
    const sidebar = document.querySelector('.sidebar');
    const overlay = document.getElementById('sidebar-overlay');
    
    if (toggleBtn && sidebar) {
      toggleBtn.addEventListener('click', () => {
        this.state.sidebarOpen = !this.state.sidebarOpen;
        sidebar.classList.toggle('open', this.state.sidebarOpen);
        if (overlay) {
          overlay.classList.toggle('active', this.state.sidebarOpen);
        }
      });
      
      // Close sidebar when clicking overlay
      if (overlay) {
        overlay.addEventListener('click', () => {
          this.state.sidebarOpen = false;
          sidebar.classList.remove('open');
          overlay.classList.remove('active');
        });
      }
      
      // Close sidebar on navigation item click (mobile)
      document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', () => {
          if (window.innerWidth <= 768) {
            this.state.sidebarOpen = false;
            sidebar.classList.remove('open');
            if (overlay) overlay.classList.remove('active');
          }
        });
      });
    }
  }
  
  initializeStreaming() {
    console.log('Initializing real-time streaming...');
    
    // Initialize SSE Event Stream
    if (typeof EventStreamClient !== 'undefined') {
      try {
        this.eventStreamClient = new EventStreamClient();
        
        this.eventStreamClient.on('connected', (data) => {
          console.log('Event stream connected:', data);
          this.updateStreamingStatus('sse', true);
        });
        
        this.eventStreamClient.on('event', (data) => {
          console.log('Real-time event received:', data);
          this.handleRealtimeEvent(data);
        });
        
        this.eventStreamClient.on('status', (data) => {
          console.log('System status update:', data);
          this.updateSystemStatus(data);
        });
        
        this.eventStreamClient.on('error', (data) => {
          console.error('Event stream error:', data);
          this.updateStreamingStatus('sse', false);
        });
        
        this.eventStreamClient.connect();
      } catch (error) {
        console.error('Failed to initialize event stream:', error);
      }
    }
    
    // Initialize WebSocket (optional)
    if (typeof io !== 'undefined' && typeof WebSocketClient !== 'undefined') {
      try {
        this.webSocketClient = new WebSocketClient();
        
        this.webSocketClient.on('connect', () => {
          console.log('WebSocket connected');
          this.updateStreamingStatus('ws', true);
        });
        
        this.webSocketClient.on('disconnect', () => {
          console.log('WebSocket disconnected');
          this.updateStreamingStatus('ws', false);
        });
        
        this.webSocketClient.on('live_event', (data) => {
          console.log('WebSocket event:', data);
          this.handleRealtimeEvent(data);
        });
        
        this.webSocketClient.connect();
      } catch (error) {
        console.error('Failed to initialize WebSocket:', error);
      }
    }
  }
  
  handleRealtimeEvent(eventData) {
    console.log('Handling real-time event:', eventData);
    
    // Refresh events list
    this.refreshEvents();
    
    // Show notification based on event type
    if (eventData.event_type) {
      if (eventData.event_type.includes('KNOWN_VISITOR')) {
        this.notificationSystem.show(
          'Known Visitor Detected',
          `${eventData.person_name || 'A known person'} is at your door`,
          'success'
        );
      } else if (eventData.event_type.includes('UNKNOWN_VISITOR')) {
        this.notificationSystem.show(
          'Unknown Visitor',
          'An unrecognized person is at your door',
          'warning'
        );
      } else if (eventData.event_type.includes('BLACKLIST_ALERT')) {
        this.notificationSystem.show(
          'Security Alert',
          'A blacklisted person has been detected!',
          'error'
        );
      } else if (eventData.event_type.includes('DOORBELL')) {
        this.notificationSystem.show(
          'Doorbell Pressed',
          'Someone rang the doorbell',
          'info'
        );
      }
    }
  }
  
  updateSystemStatus(statusData) {
    if (statusData.streaming_status === 'running') {
      this.state.systemOnline = true;
      const statusEl = document.getElementById('system-status');
      if (statusEl) {
        statusEl.className = 'status-indicator';
        statusEl.setAttribute('data-status', 'online');
      }
    }
  }
  
  updateStreamingStatus(type, connected) {
    console.log(`Streaming ${type} status:`, connected ? 'connected' : 'disconnected');
    // Could update UI indicator here
  }
  
  async refreshAll() {
    console.log('Refreshing all data...');
    const refreshBtn = document.getElementById('refresh-data');
    if (refreshBtn) {
      refreshBtn.classList.add('loading');
    }
    
    try {
      await Promise.all([
        this.refreshStatus(),
        this.refreshEvents(),
        this.refreshKnownFaces()
      ]);
      
      this.notificationSystem.show('Refreshed', 'Dashboard data updated', 'success', 2000);
    } catch (error) {
      console.error('Error refreshing data:', error);
      this.notificationSystem.show('Error', 'Failed to refresh data', 'error');
    } finally {
      if (refreshBtn) {
        refreshBtn.classList.remove('loading');
      }
    }
  }
  
  async refreshStatus() {
    try {
      const result = await this.apiCall('status');
      
      if (result.status === 'online') {
        this.state.systemOnline = true;
        this.state.lastUpdate = new Date();
        
        // Update status indicator
        const statusEl = document.getElementById('system-status');
        const statusText = document.getElementById('status-text');
        
        if (statusEl) {
          statusEl.className = 'status-indicator';
          statusEl.setAttribute('data-status', 'online');
        }
        if (statusText) {
          statusText.textContent = 'System Online';
        }
        
        // Update component statuses
        this.updateStatusField('camera-status', 
          result.camera.initialized ? `✅ ${result.camera.camera_type}` : '❌ Not initialized');
        
        this.updateStatusField('gpio-status', 
          result.gpio.initialized ? '✅ Ready' : '❌ Not initialized');
        
        this.updateStatusField('telegram-status', 
          result.telegram.initialized ? '✅ Connected' : '❌ Not configured');
        
        this.updateStatusField('known-faces', 
          `${result.faces.known_faces} known, ${result.faces.blacklist_faces} blacklisted`);
        
        // Update LED states
        if (result.gpio.led_states) {
          this.updateStatusField('red-led-status', 
            result.gpio.led_states.red ? '🔴 ON' : '⚫ OFF');
          this.updateStatusField('yellow-led-status', 
            result.gpio.led_states.yellow ? '🟡 ON' : '⚫ OFF');
          this.updateStatusField('green-led-status', 
            result.gpio.led_states.green ? '🟢 ON' : '⚫ OFF');
        }
      } else {
        this.state.systemOnline = false;
        const statusEl = document.getElementById('system-status');
        const statusText = document.getElementById('status-text');
        
        if (statusEl) {
          statusEl.className = 'status-indicator';
          statusEl.setAttribute('data-status', 'offline');
        }
        if (statusText) {
          statusText.textContent = 'System Offline';
        }
      }
      
      // Update last updated time
      const lastUpdated = document.getElementById('last-updated');
      if (lastUpdated) {
        lastUpdated.textContent = new Date().toLocaleTimeString();
      }
    } catch (error) {
      console.error('Error refreshing status:', error);
    }
  }
  
  async refreshEvents() {
    try {
      const result = await this.apiCall('recent-events');
      
      if (result.status === 'success') {
        const container = document.getElementById('recent-events') || document.getElementById('events-list');
        
        if (!container) return;
        
        if (result.events.length === 0) {
          container.innerHTML = '<p style="color: var(--gray-500);">No recent events</p>';
          return;
        }
        
        let html = '';
        result.events.reverse().forEach(event => {
          const timestamp = new Date(event.timestamp).toLocaleString();
          const icon = this.getEventIcon(event.event_type);
          const className = this.getEventClassName(event.event_type);
          
          html += `
            <div class="event-item ${className}">
              <strong>${icon} ${event.event_type.replace(/_/g, ' ')}</strong>
              <span class="event-time">${timestamp}</span>
              ${event.faces && event.faces.length > 0 ? 
                `<div style="margin-top: var(--space-2); font-size: var(--text-sm);">
                  Faces: ${event.faces.map(f => f.name || f.status).join(', ')}
                </div>` : 
                ''}
            </div>
          `;
        });
        
        container.innerHTML = html;
      }
    } catch (error) {
      console.error('Error refreshing events:', error);
    }
  }
  
  async refreshKnownFaces() {
    // This function can be called if needed
    console.log('Refreshing known faces...');
  }
  
  getEventIcon(eventType) {
    if (eventType.includes('KNOWN_VISITOR')) return '✅';
    if (eventType.includes('UNKNOWN_VISITOR')) return '❓';
    if (eventType.includes('BLACKLIST_ALERT')) return '🚨';
    if (eventType.includes('DOORBELL')) return '🔔';
    return '📋';
  }
  
  getEventClassName(eventType) {
    if (eventType.includes('KNOWN_VISITOR')) return 'known';
    if (eventType.includes('UNKNOWN_VISITOR')) return 'unknown';
    if (eventType.includes('BLACKLIST_ALERT')) return 'alert';
    return '';
  }
  
  updateStatusField(elementId, value) {
    const element = document.getElementById(elementId);
    if (element) {
      element.textContent = value;
    }
  }
  
  async apiCall(endpoint, method = 'GET', data = null) {
    try {
      const options = {
        method: method,
        headers: {
          'Content-Type': 'application/json',
        }
      };
      
      if (data) {
        options.body = JSON.stringify(data);
      }
      
      const response = await fetch(`/api/${endpoint}`, options);
      return await response.json();
    } catch (error) {
      console.error('API call failed:', error);
      return { status: 'error', message: error.message };
    }
  }
  
  requestNotificationPermission() {
    if ('Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission().then(permission => {
        if (permission === 'granted') {
          console.log('Notification permission granted');
        }
      });
    }
  }
  
  cleanup() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
    }
    if (this.eventStreamClient) {
      this.eventStreamClient.disconnect();
    }
    if (this.webSocketClient) {
      this.webSocketClient.disconnect();
    }
  }
}

/**
 * Notification System
 * 
 * Manages toast notifications
 */
class NotificationSystem {
  constructor() {
    this.container = this.createContainer();
    this.notifications = [];
  }
  
  createContainer() {
    let container = document.getElementById('notification-container');
    if (!container) {
      container = document.createElement('div');
      container.id = 'notification-container';
      container.style.position = 'fixed';
      container.style.top = 'var(--space-4)';
      container.style.right = 'var(--space-4)';
      container.style.zIndex = 'var(--z-tooltip)';
      document.body.appendChild(container);
    }
    return container;
  }
  
  show(title, message, type = 'info', duration = 5000) {
    const notification = this.createNotification(title, message, type);
    this.container.appendChild(notification);
    
    // Trigger show animation
    setTimeout(() => {
      notification.classList.add('show');
    }, 10);
    
    // Auto-dismiss
    if (duration > 0) {
      setTimeout(() => {
        this.dismiss(notification);
      }, duration);
    }
    
    // Show browser notification if permitted
    this.showBrowserNotification(title, message);
    
    return notification;
  }
  
  createNotification(title, message, type) {
    const notification = document.createElement('div');
    notification.className = `notification ${type}`;
    
    const icon = this.getIcon(type);
    
    notification.innerHTML = `
      <div class="notification-icon">${icon}</div>
      <div class="notification-content">
        <div class="notification-title">${this.escapeHtml(title)}</div>
        <div class="notification-message">${this.escapeHtml(message)}</div>
      </div>
      <button class="notification-close" aria-label="Close notification">×</button>
    `;
    
    // Close button handler
    const closeBtn = notification.querySelector('.notification-close');
    closeBtn.addEventListener('click', () => {
      this.dismiss(notification);
    });
    
    return notification;
  }
  
  dismiss(notification) {
    notification.classList.remove('show');
    notification.classList.add('hide');
    
    setTimeout(() => {
      if (notification.parentNode) {
        notification.parentNode.removeChild(notification);
      }
    }, 300);
  }
  
  getIcon(type) {
    const icons = {
      success: '✓',
      error: '✕',
      warning: '⚠',
      info: 'ℹ'
    };
    return icons[type] || icons.info;
  }
  
  showBrowserNotification(title, message) {
    if ('Notification' in window && Notification.permission === 'granted') {
      try {
        new Notification(title, { 
          body: message,
          icon: '/static/images/icon-192.png'
        });
      } catch (error) {
        console.error('Failed to show browser notification:', error);
      }
    }
  }
  
  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }
}

// Initialize dashboard when DOM is ready
let dashboardInstance = null;

document.addEventListener('DOMContentLoaded', () => {
  dashboardInstance = new Dashboard();
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  if (dashboardInstance) {
    dashboardInstance.cleanup();
  }
});

// Export for use in other scripts
window.Dashboard = Dashboard;
window.NotificationSystem = NotificationSystem;
