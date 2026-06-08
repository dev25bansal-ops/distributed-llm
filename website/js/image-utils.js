/**
 * Image utility — handles modern image formats with fallbacks.
 *
 * Usage:
 *   import { createPicture } from './js/image-utils.js';
 *
 *   const picture = createPicture({
 *     src: '/images/hero.png',
 *     alt: 'DistLLM Architecture',
 *     width: 800,
 *     height: 400,
 *   });
 *   document.body.appendChild(picture);
 */

/**
 * Create a <picture> element with WebP/AVIF fallbacks.
 * @param {Object} options
 * @param {string} options.src - Source image path (PNG/JPEG)
 * @param {string} options.alt - Alt text for accessibility
 * @param {number} [options.width] - Image width
 * @param {number} [options.height] - Image height
 * @param {string} [options.sizes] - Sizes attribute for responsive images
 * @param {string} [options.loading] - Loading attribute (lazy/eager)
 * @returns {HTMLPictureElement}
 */
export function createPicture({ src, alt, width, height, sizes, loading = 'lazy' }) {
    const picture = document.createElement('picture');

    // Get base path and extension
    const basePath = src.replace(/\.[^/.]+$/, '');
    const ext = src.split('.').pop().toLowerCase();

    // Only add modern formats for raster images
    const isRaster = ['png', 'jpg', 'jpeg', 'gif'].includes(ext);

    if (isRaster) {
        // AVIF source (best compression)
        const avifSource = document.createElement('source');
        avifSource.srcset = `${basePath}.avif`;
        avifSource.type = 'image/avif';
        if (sizes) avifSource.sizes = sizes;
        picture.appendChild(avifSource);

        // WebP source (good compression, wide support)
        const webpSource = document.createElement('source');
        webpSource.srcset = `${basePath}.webp`;
        webpSource.type = 'image/webp';
        if (sizes) webpSource.sizes = sizes;
        picture.appendChild(webpSource);
    }

    // Fallback <img>
    const img = document.createElement('img');
    img.src = src;
    img.alt = alt || '';
    if (width) img.width = width;
    if (height) img.height = height;
    if (loading) img.loading = loading;
    img.decoding = 'async';

    picture.appendChild(img);

    return picture;
}

/**
 * Create a responsive image with srcset.
 * @param {Object} options
 * @param {string} options.src - Base image path
 * @param {string} options.alt - Alt text
 * @param {number[]} options.widths - Array of widths for srcset
 * @param {string} [options.sizes] - Sizes attribute
 * @param {number} [options.width] - Default width
 * @param {number} [options.height] - Default height
 * @param {string} [options.loading] - Loading attribute
 * @returns {HTMLImageElement}
 */
export function createResponsiveImage({ src, alt, widths, sizes, width, height, loading = 'lazy' }) {
    const img = document.createElement('img');

    // Build srcset
    const basePath = src.replace(/\.[^/.]+$/, '');
    const ext = src.split('.').pop();
    const srcset = widths.map(w => `${basePath}-${w}w.${ext} ${w}w`).join(', ');

    img.srcset = srcset;
    if (sizes) {
        img.sizes = sizes;
    } else {
        // Default sizes based on viewport
        img.sizes = '(max-width: 768px) 100vw, (max-width: 1200px) 50vw, 33vw';
    }

    img.src = src; // Fallback
    img.alt = alt || '';
    if (width) img.width = width;
    if (height) img.height = height;
    if (loading) img.loading = loading;
    img.decoding = 'async';

    return img;
}

/**
 * Lazy load an image when it enters the viewport.
 * @param {HTMLImageElement} img - Image element to lazy load
 * @param {string} src - Image source URL
 * @param {string} [srcset] - Srcset attribute
 * @param {string} [sizes] - Sizes attribute
 */
export function lazyLoadImage(img, src, srcset, sizes) {
    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    const target = entry.target;
                    target.src = src;
                    if (srcset) target.srcset = srcset;
                    if (sizes) target.sizes = sizes;
                    target.classList.add('loaded');
                    observer.unobserve(target);
                }
            });
        }, {
            rootMargin: '200px 0px',
            threshold: 0
        });

        observer.observe(img);
    } else {
        // Fallback: load immediately
        img.src = src;
        if (srcset) img.srcset = srcset;
        if (sizes) img.sizes = sizes;
    }
}

/**
 * Preload critical images.
 * @param {string[]} urls - Array of image URLs to preload
 */
export function preloadImages(urls) {
    urls.forEach(url => {
        const link = document.createElement('link');
        link.rel = 'preload';
        link.as = 'image';
        link.href = url;

        // Add type hint for modern formats
        if (url.endsWith('.webp')) {
            link.type = 'image/webp';
        } else if (url.endsWith('.avif')) {
            link.type = 'image/avif';
        }

        document.head.appendChild(link);
    });
}
