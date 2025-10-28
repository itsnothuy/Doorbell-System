/**
 * Server-Sent Events (SSE) Client
 * 
 * Handles real-time event streaming from the doorbell security system.
 */

class EventStreamClient {
    constructor(eventUrl = '/stream/events', statusUrl = '/stream/system-status') {
        this.eventUrl = eventUrl;
        this.statusUrl = statusUrl;
        this.clientId = this.generateClientId();
        this.eventSource = null;
        this.statusSource = null;
        this.eventHandlers = {};
        this.reconnectDelay = 3000;
        this.maxReconnectDelay = 30000;
        this.reconnectAttempts = 0;
    }

    generateClientId() {
        return 'client_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }

    connect() {
        this.connectEvents();
        this.connectStatus();
    }

    connectEvents() {
        if (this.eventSource) {
            this.eventSource.close();
        }

        const url = `${this.eventUrl}?client_id=${this.clientId}`;
        console.log('Connecting to event stream:', url);

        this.eventSource = new EventSource(url);

        this.eventSource.addEventListener('connected', (e) => {
            const data = JSON.parse(e.data);
            console.log('Connected to event stream:', data);
            this.reconnectAttempts = 0;
            this.trigger('connected', data);
        });

        this.eventSource.addEventListener('event', (e) => {
            const data = JSON.parse(e.data);
            console.log('Event received:', data);
            this.trigger('event', data);
        });

        this.eventSource.addEventListener('heartbeat', (e) => {
            const data = JSON.parse(e.data);
            console.log('Heartbeat:', data.timestamp);
        });

        this.eventSource.addEventListener('error', (e) => {
            if (e.data) {
                const data = JSON.parse(e.data);
                console.error('Stream error:', data);
                this.trigger('error', data);
            }
        });

        this.eventSource.onerror = (error) => {
            console.error('EventSource error:', error);
            this.reconnect('events');
        };
    }

    connectStatus() {
        if (this.statusSource) {
            this.statusSource.close();
        }

        const url = `${this.statusUrl}?client_id=${this.clientId}`;
        console.log('Connecting to status stream:', url);

        this.statusSource = new EventSource(url);

        this.statusSource.addEventListener('status_connected', (e) => {
            const data = JSON.parse(e.data);
            console.log('Connected to status stream:', data);
        });

        this.statusSource.addEventListener('system_status', (e) => {
            const data = JSON.parse(e.data);
            this.trigger('status', data);
        });

        this.statusSource.onerror = (error) => {
            console.error('StatusSource error:', error);
            this.reconnect('status');
        };
    }

    reconnect(type) {
        const delay = Math.min(
            this.reconnectDelay * Math.pow(2, this.reconnectAttempts),
            this.maxReconnectDelay
        );

        console.log(`Reconnecting ${type} in ${delay}ms...`);
        this.reconnectAttempts++;

        setTimeout(() => {
            if (type === 'events') {
                this.connectEvents();
            } else {
                this.connectStatus();
            }
        }, delay);
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
        if (this.eventSource) {
            this.eventSource.close();
            this.eventSource = null;
        }
        if (this.statusSource) {
            this.statusSource.close();
            this.statusSource = null;
        }
        console.log('Disconnected from event streams');
    }
}

// Export for use in other scripts
window.EventStreamClient = EventStreamClient;
