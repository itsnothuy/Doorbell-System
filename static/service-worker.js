/**
 * Doorbell Security System - Service Worker
 * 
 * Provides offline support and caching for PWA functionality
 */

const CACHE_NAME = 'doorbell-security-v1.0.0';
const STATIC_CACHE_URLS = [
  '/',
  '/static/css/design-system.css',
  '/static/css/components.css',
  '/static/css/layout.css',
  '/static/js/dashboard.js',
  '/static/js/event-stream.js',
  '/static/js/websocket-client.js',
  '/static/js/video-stream.js',
  '/static/manifest.json'
];

// Install event - cache static assets
self.addEventListener('install', event => {
  console.log('[Service Worker] Installing...');
  
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => {
        console.log('[Service Worker] Caching static assets');
        return cache.addAll(STATIC_CACHE_URLS);
      })
      .then(() => {
        console.log('[Service Worker] Installation complete');
        return self.skipWaiting();
      })
      .catch(error => {
        console.error('[Service Worker] Installation failed:', error);
      })
  );
});

// Activate event - cleanup old caches
self.addEventListener('activate', event => {
  console.log('[Service Worker] Activating...');
  
  event.waitUntil(
    caches.keys()
      .then(cacheNames => {
        return Promise.all(
          cacheNames
            .filter(cacheName => cacheName !== CACHE_NAME)
            .map(cacheName => {
              console.log('[Service Worker] Deleting old cache:', cacheName);
              return caches.delete(cacheName);
            })
        );
      })
      .then(() => {
        console.log('[Service Worker] Activation complete');
        return self.clients.claim();
      })
  );
});

// Fetch event - serve from cache with network fallback
self.addEventListener('fetch', event => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-HTTP requests
  if (!url.protocol.startsWith('http')) {
    return;
  }
  
  // Handle API requests - network first, cache fallback
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(request)
        .then(response => {
          // Only cache successful GET requests
          if (request.method === 'GET' && response.ok) {
            const responseClone = response.clone();
            caches.open(CACHE_NAME)
              .then(cache => cache.put(request, responseClone))
              .catch(error => console.error('[Service Worker] Cache put failed:', error));
          }
          return response;
        })
        .catch(() => {
          // Try to serve from cache
          return caches.match(request)
            .then(cachedResponse => {
              if (cachedResponse) {
                console.log('[Service Worker] Serving API from cache:', url.pathname);
                return cachedResponse;
              }
              // Return offline response
              return new Response(
                JSON.stringify({ 
                  status: 'offline', 
                  message: 'You are currently offline. Some features may not be available.' 
                }),
                { 
                  headers: { 'Content-Type': 'application/json' },
                  status: 503
                }
              );
            });
        })
    );
    return;
  }
  
  // Handle streaming endpoints - always use network
  if (url.pathname.startsWith('/stream/')) {
    event.respondWith(fetch(request));
    return;
  }
  
  // Handle static assets - cache first, network fallback
  event.respondWith(
    caches.match(request)
      .then(cachedResponse => {
        if (cachedResponse) {
          console.log('[Service Worker] Serving from cache:', url.pathname);
          return cachedResponse;
        }
        
        console.log('[Service Worker] Fetching from network:', url.pathname);
        return fetch(request)
          .then(response => {
            // Cache successful responses for future use
            if (response.ok && request.method === 'GET') {
              const responseClone = response.clone();
              caches.open(CACHE_NAME)
                .then(cache => cache.put(request, responseClone))
                .catch(error => console.error('[Service Worker] Cache put failed:', error));
            }
            return response;
          })
          .catch(error => {
            console.error('[Service Worker] Fetch failed:', error);
            
            // Return offline page for navigation requests
            if (request.mode === 'navigate') {
              return caches.match('/')
                .then(cachedIndex => {
                  if (cachedIndex) {
                    return cachedIndex;
                  }
                  return new Response(
                    '<h1>Offline</h1><p>You are currently offline. Please check your internet connection.</p>',
                    { headers: { 'Content-Type': 'text/html' } }
                  );
                });
            }
            
            throw error;
          });
      })
  );
});

// Background sync for offline actions
self.addEventListener('sync', event => {
  console.log('[Service Worker] Background sync:', event.tag);
  
  if (event.tag === 'sync-events') {
    event.waitUntil(syncEvents());
  }
});

// Push notifications
self.addEventListener('push', event => {
  console.log('[Service Worker] Push notification received');
  
  const options = {
    body: 'Someone is at your door',
    icon: '/static/images/icon-192.png',
    badge: '/static/images/badge-72.png',
    vibrate: [200, 100, 200],
    data: {
      dateOfArrival: Date.now(),
      primaryKey: 1
    },
    actions: [
      {
        action: 'view',
        title: 'View Dashboard',
        icon: '/static/images/view-icon.png'
      },
      {
        action: 'close',
        title: 'Dismiss',
        icon: '/static/images/close-icon.png'
      }
    ]
  };
  
  if (event.data) {
    const data = event.data.json();
    options.body = data.message || options.body;
    options.data = { ...options.data, ...data };
  }
  
  event.waitUntil(
    self.registration.showNotification('Doorbell Security System', options)
  );
});

// Handle notification clicks
self.addEventListener('notificationclick', event => {
  console.log('[Service Worker] Notification click:', event.action);
  
  event.notification.close();
  
  if (event.action === 'view') {
    event.waitUntil(
      clients.openWindow('/')
    );
  }
});

// Helper function to sync events when back online
async function syncEvents() {
  try {
    // Get pending events from IndexedDB or localStorage
    const pendingEvents = await getPendingEvents();
    
    if (pendingEvents.length === 0) {
      console.log('[Service Worker] No pending events to sync');
      return;
    }
    
    console.log(`[Service Worker] Syncing ${pendingEvents.length} pending events`);
    
    // Send pending events to server
    for (const event of pendingEvents) {
      try {
        const response = await fetch('/api/sync-event', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(event)
        });
        
        if (response.ok) {
          await removePendingEvent(event.id);
          console.log('[Service Worker] Event synced:', event.id);
        }
      } catch (error) {
        console.error('[Service Worker] Failed to sync event:', error);
      }
    }
  } catch (error) {
    console.error('[Service Worker] Sync failed:', error);
    throw error;
  }
}

// Placeholder functions for IndexedDB integration
async function getPendingEvents() {
  // TODO: Implement IndexedDB retrieval
  return [];
}

async function removePendingEvent(eventId) {
  // TODO: Implement IndexedDB removal
  return true;
}

// Log service worker status
console.log('[Service Worker] Loaded');
