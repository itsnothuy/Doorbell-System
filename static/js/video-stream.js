/**
 * Video Stream Client
 * 
 * Handles live video streaming display from the doorbell camera.
 */

class VideoStreamClient {
    constructor(videoElement) {
        this.videoElement = videoElement;
        this.streamUrl = null;
        this.clientId = this.generateClientId();
        this.quality = 'medium';
        this.isStreaming = false;
    }

    generateClientId() {
        return 'video_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }

    async startStream(quality = 'medium') {
        if (this.isStreaming) {
            console.log('Stream already active');
            return;
        }

        this.quality = quality;
        this.streamUrl = `/stream/video/${this.clientId}?quality=${quality}`;
        
        console.log('Starting video stream:', this.streamUrl);

        try {
            // Test if stream is available
            const response = await fetch(this.streamUrl, { method: 'HEAD' });
            if (!response.ok) {
                throw new Error('Video stream not available');
            }

            // Set video source
            this.videoElement.src = this.streamUrl;
            this.isStreaming = true;

            // Handle load events
            this.videoElement.onload = () => {
                console.log('Video stream loaded');
            };

            this.videoElement.onerror = (error) => {
                console.error('Video stream error:', error);
                this.isStreaming = false;
            };

        } catch (error) {
            console.error('Failed to start video stream:', error);
            this.isStreaming = false;
            throw error;
        }
    }

    stopStream() {
        if (!this.isStreaming) {
            return;
        }

        console.log('Stopping video stream');
        this.videoElement.src = '';
        this.isStreaming = false;
        this.streamUrl = null;
    }

    changeQuality(quality) {
        if (!this.isStreaming) {
            console.warn('No active stream to change quality');
            return;
        }

        console.log('Changing video quality to:', quality);
        this.stopStream();
        setTimeout(() => {
            this.startStream(quality);
        }, 100);
    }

    getStatus() {
        return {
            isStreaming: this.isStreaming,
            quality: this.quality,
            streamUrl: this.streamUrl,
            clientId: this.clientId
        };
    }
}

// Export for use in other scripts
window.VideoStreamClient = VideoStreamClient;
