/**
 * WebSocket Client
 * 
 * Provides bidirectional real-time communication with the doorbell security system.
 */

class WebSocketClient {
    constructor(url = null) {
        // Auto-detect WebSocket URL
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const host = window.location.host;
        this.url = url || `${protocol}//${host}/socket.io/`;
        
        this.socket = null;
        this.connected = false;
        this.eventHandlers = {};
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.reconnectDelay = 3000;
    }

    connect() {
        if (this.socket) {
            this.disconnect();
        }

        console.log('Connecting to WebSocket:', this.url);

        // Using Socket.IO client (needs to be included in HTML)
        if (typeof io === 'undefined') {
            console.error('Socket.IO client not loaded. Please include socket.io.js');
            return;
        }

        this.socket = io(this.url, {
            transports: ['websocket', 'polling'],
            reconnection: true,
            reconnectionDelay: this.reconnectDelay,
            reconnectionAttempts: this.maxReconnectAttempts
        });

        this.setupEventHandlers();
    }

    setupEventHandlers() {
        this.socket.on('connect', () => {
            console.log('WebSocket connected');
            this.connected = true;
            this.reconnectAttempts = 0;
            this.trigger('connect', { connected: true });
        });

        this.socket.on('disconnect', (reason) => {
            console.log('WebSocket disconnected:', reason);
            this.connected = false;
            this.trigger('disconnect', { reason });
        });

        this.socket.on('connected', (data) => {
            console.log('Server acknowledged connection:', data);
            this.trigger('server_connected', data);
        });

        this.socket.on('live_event', (data) => {
            console.log('Live event received:', data);
            this.trigger('live_event', data);
        });

        this.socket.on('command_result', (data) => {
            console.log('Command result:', data);
            this.trigger('command_result', data);
        });

        this.socket.on('video_stream_ready', (data) => {
            console.log('Video stream ready:', data);
            this.trigger('video_stream_ready', data);
        });

        this.socket.on('error', (data) => {
            console.error('WebSocket error:', data);
            this.trigger('error', data);
        });

        this.socket.on('connect_error', (error) => {
            console.error('Connection error:', error);
            this.reconnectAttempts++;
            if (this.reconnectAttempts >= this.maxReconnectAttempts) {
                console.error('Max reconnection attempts reached');
                this.trigger('max_reconnect', { attempts: this.reconnectAttempts });
            }
        });
    }

    subscribe(eventTypes) {
        if (!this.connected) {
            console.warn('Not connected, cannot subscribe');
            return;
        }

        console.log('Subscribing to event types:', eventTypes);
        this.socket.emit('subscribe', { event_types: eventTypes });
    }

    sendCommand(command, params = {}) {
        if (!this.connected) {
            console.warn('Not connected, cannot send command');
            return Promise.reject(new Error('Not connected'));
        }

        return new Promise((resolve, reject) => {
            console.log('Sending command:', command, params);
            
            // Set up one-time listener for command result
            const handler = (data) => {
                if (data.command === command) {
                    this.off('command_result', handler);
                    if (data.success) {
                        resolve(data.result);
                    } else {
                        reject(new Error(data.error));
                    }
                }
            };
            
            this.on('command_result', handler);
            this.socket.emit('system_command', { command, params });
            
            // Timeout after 30 seconds
            setTimeout(() => {
                this.off('command_result', handler);
                reject(new Error('Command timeout'));
            }, 30000);
        });
    }

    requestVideoStream(quality = 'medium') {
        if (!this.connected) {
            console.warn('Not connected, cannot request video stream');
            return Promise.reject(new Error('Not connected'));
        }

        return new Promise((resolve, reject) => {
            console.log('Requesting video stream:', quality);
            
            const handler = (data) => {
                this.off('video_stream_ready', handler);
                resolve(data);
            };
            
            this.on('video_stream_ready', handler);
            this.socket.emit('request_video_stream', { quality });
            
            setTimeout(() => {
                this.off('video_stream_ready', handler);
                reject(new Error('Video stream request timeout'));
            }, 10000);
        });
    }

    on(event, handler) {
        if (!this.eventHandlers[event]) {
            this.eventHandlers[event] = [];
        }
        this.eventHandlers[event].push(handler);
    }

    off(event, handler) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event] = this.eventHandlers[event].filter(h => h !== handler);
        }
    }

    trigger(event, data) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error('Event handler error:', error);
                }
            });
        }
    }

    disconnect() {
        if (this.socket) {
            this.socket.disconnect();
            this.socket = null;
            this.connected = false;
            console.log('Disconnected from WebSocket');
        }
    }
}

// Export for use in other scripts
window.WebSocketClient = WebSocketClient;
