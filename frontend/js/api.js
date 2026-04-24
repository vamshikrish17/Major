/**
 * VisionExtract 2.0 — API Client
 * Handles all communication with the FastAPI backend.
 */

const API_BASE = '/api';

const VisionAPI = {
    /**
     * Analyze an image file.
     * @param {File} file - Image file to analyze
     * @param {string} mode - Analysis mode (general, bio, space)
     * @param {number} confidence - Confidence threshold (0-1)
     * @param {function} onProgress - Progress callback
     * @returns {Promise<Object>} Analysis response
     */
    async analyzeImage(file, mode = 'general', confidence = 0.35, onProgress = null) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('mode', mode);
        formData.append('confidence', confidence.toString());
        formData.append('use_hybrid_prompts', 'true');

        return this._upload(`${API_BASE}/analyze`, formData, onProgress);
    },

    /**
     * Analyze a webcam capture (base64).
     * @param {string} base64Data - Base64-encoded image data
     * @param {string} mode - Analysis mode
     * @param {number} confidence - Confidence threshold
     * @returns {Promise<Object>} Analysis response
     */
    async analyzeWebcam(base64Data, mode = 'general', confidence = 0.35) {
        const formData = new FormData();
        formData.append('webcam_data', base64Data);
        formData.append('mode', mode);
        formData.append('confidence', confidence.toString());

        const response = await fetch(`${API_BASE}/analyze`, {
            method: 'POST',
            body: formData,
        });

        if (!response.ok) {
            const err = await response.json().catch(() => ({}));
            throw new Error(err.detail || `Analysis failed (${response.status})`);
        }

        return response.json();
    },

    /**
     * Get analysis history.
     * @param {number} page
     * @param {number} perPage
     * @returns {Promise<Object>}
     */
    async getResults(page = 1, perPage = 10, mode = null) {
        const params = new URLSearchParams({
            page: page.toString(),
            per_page: perPage.toString(),
        });
        if (mode) params.set('mode', mode);
        const res = await fetch(`${API_BASE}/results?${params.toString()}`);
        if (!res.ok) throw new Error('Failed to fetch results');
        return res.json();
    },

    /**
     * Get a specific analysis result.
     * @param {string} id - Analysis ID
     * @returns {Promise<Object>}
     */
    async getResult(id) {
        const res = await fetch(`${API_BASE}/results/${id}`);
        if (!res.ok) throw new Error('Result not found');
        return res.json();
    },

    /**
     * Delete an analysis result.
     * @param {string} id
     * @returns {Promise<Object>}
     */
    async deleteResult(id) {
        const res = await fetch(`${API_BASE}/results/${id}`, { method: 'DELETE' });
        if (!res.ok) throw new Error('Failed to delete');
        return res.json();
    },

    /**
     * Check system health.
     * @returns {Promise<Object>}
     */
    async getHealth() {
        const res = await fetch(`${API_BASE}/health`);
        if (!res.ok) throw new Error('Health check failed');
        return res.json();
    },

    /**
     * Upload with progress tracking.
     */
    _upload(url, formData, onProgress) {
        return new Promise((resolve, reject) => {
            const xhr = new XMLHttpRequest();

            xhr.upload.addEventListener('progress', (e) => {
                if (e.lengthComputable && onProgress) {
                    onProgress(Math.round((e.loaded / e.total) * 100));
                }
            });

            xhr.addEventListener('load', () => {
                if (xhr.status >= 200 && xhr.status < 300) {
                    try {
                        resolve(JSON.parse(xhr.responseText));
                    } catch {
                        reject(new Error('Invalid response'));
                    }
                } else {
                    try {
                        const err = JSON.parse(xhr.responseText);
                        reject(new Error(err.detail || `Upload failed (${xhr.status})`));
                    } catch {
                        reject(new Error(`Upload failed (${xhr.status})`));
                    }
                }
            });

            xhr.addEventListener('error', () => reject(new Error('Network error')));
            xhr.addEventListener('abort', () => reject(new Error('Upload cancelled')));

            xhr.open('POST', url);
            xhr.send(formData);
        });
    }
};
