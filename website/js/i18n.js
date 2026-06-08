/**
 * Internationalization (i18n) System
 *
 * Features:
 * - JSON translation files
 * - Language switcher component
 * - Automatic language detection
 * - LocalStorage persistence
 * - Dynamic content translation
 *
 * Usage:
 *   import { initI18n, t, setLanguage } from './js/i18n.js';
 *   initI18n();
 *   const text = t('hero.title');
 */

// ── Configuration ──────────────────────────────────────────────────────

const SUPPORTED_LANGUAGES = [
    { code: 'en', name: 'English', nativeName: 'English', flag: '🇺🇸' },
    { code: 'zh-CN', name: 'Chinese (Simplified)', nativeName: '简体中文', flag: '🇨🇳' },
    { code: 'ja-JP', name: 'Japanese', nativeName: '日本語', flag: '🇯🇵' },
    { code: 'ko-KR', name: 'Korean', nativeName: '한국어', flag: '🇰🇷' },
];

const DEFAULT_LANGUAGE = 'en';
const STORAGE_KEY = 'distllm-language';

// ── State ──────────────────────────────────────────────────────────────

let currentLanguage = DEFAULT_LANGUAGE;
let translations = {};
let isInitialized = false;

// ── Translation Loading ────────────────────────────────────────────────

async function loadTranslations(lang) {
    try {
        const response = await fetch(`/locales/${lang}.json`);
        if (!response.ok) throw new Error(`Failed to load ${lang}`);
        translations[lang] = await response.json();
        return translations[lang];
    } catch (e) {
        console.warn(`[i18n] Failed to load translations for ${lang}:`, e);
        if (lang !== DEFAULT_LANGUAGE) {
            return loadTranslations(DEFAULT_LANGUAGE);
        }
        return {};
    }
}

// ── Language Detection ─────────────────────────────────────────────────

function detectLanguage() {
    // Check localStorage first
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored && SUPPORTED_LANGUAGES.some(l => l.code === stored)) {
        return stored;
    }

    // Check browser language
    const browserLang = navigator.language || navigator.userLanguage;
    if (browserLang) {
        // Try exact match
        if (SUPPORTED_LANGUAGES.some(l => l.code === browserLang)) {
            return browserLang;
        }
        // Try language prefix (e.g., 'zh' -> 'zh-CN')
        const prefix = browserLang.split('-')[0];
        const match = SUPPORTED_LANGUAGES.find(l => l.code.startsWith(prefix));
        if (match) {
            return match.code;
        }
    }

    return DEFAULT_LANGUAGE;
}

// ── Translation Function ───────────────────────────────────────────────

export function t(key, defaultValue = '') {
    const keys = key.split('.');
    let value = translations[currentLanguage];

    for (const k of keys) {
        if (value && typeof value === 'object' && k in value) {
            value = value[k];
        } else {
            // Fallback to default language
            let fallback = translations[DEFAULT_LANGUAGE];
            for (const fk of keys) {
                if (fallback && typeof fallback === 'object' && fk in fallback) {
                    fallback = fallback[fk];
                } else {
                    return defaultValue || key;
                }
            }
            return typeof fallback === 'string' ? fallback : defaultValue || key;
        }
    }

    return typeof value === 'string' ? value : defaultValue || key;
}

// ── Language Switching ─────────────────────────────────────────────────

export async function setLanguage(lang) {
    if (!SUPPORTED_LANGUAGES.some(l => l.code === lang)) {
        console.warn(`[i18n] Unsupported language: ${lang}`);
        return;
    }

    // Load translations if not cached
    if (!translations[lang]) {
        await loadTranslations(lang);
    }

    currentLanguage = lang;
    localStorage.setItem(STORAGE_KEY, lang);

    // Update HTML lang attribute
    document.documentElement.lang = lang;

    // Update all translatable elements
    updateTranslatableElements();

    // Dispatch event for other components
    window.dispatchEvent(new CustomEvent('languageChanged', { detail: { language: lang } }));

    // console.log removed for production
}


export function getCurrentLanguage() {
    return currentLanguage;
}

export function getSupportedLanguages() {
    return SUPPORTED_LANGUAGES;
}

// ── DOM Updates ────────────────────────────────────────────────────────

function updateTranslatableElements() {
    // Update elements with data-i18n attribute
    document.querySelectorAll('[data-i18n]').forEach(el => {
        const key = el.dataset.i18n;
        const translation = t(key);
        if (translation) {
            if (el.tagName === 'INPUT' && el.type !== 'submit') {
                el.placeholder = translation;
            } else {
                el.textContent = translation;
            }
        }
    });

    // Update elements with data-i18n-title attribute
    document.querySelectorAll('[data-i18n-title]').forEach(el => {
        const key = el.dataset.i18nTitle;
        const translation = t(key);
        if (translation) {
            el.title = translation;
        }
    });

    // Update elements with data-i18n-aria-label attribute
    document.querySelectorAll('[data-i18n-aria-label]').forEach(el => {
        const key = el.dataset.i18nAriaLabel;
        const translation = t(key);
        if (translation) {
            el.setAttribute('aria-label', translation);
        }
    });

    // Update meta tags
    const metaTitle = document.querySelector('title[data-i18n]');
    if (metaTitle) {
        metaTitle.textContent = t(metaTitle.dataset.i18n);
    }

    const metaDesc = document.querySelector('meta[name="description"][data-i18n]');
    if (metaDesc) {
        metaDesc.content = t(metaDesc.dataset.i18n);
    }
}

// ── Language Switcher Component ─────────────────────────────────────────

export function createLanguageSwitcher(containerId = 'languageSwitcher') {
    const container = document.getElementById(containerId);
    if (!container) return;

    const currentLang = SUPPORTED_LANGUAGES.find(l => l.code === currentLanguage);

    container.innerHTML = `
        <div class="lang-switcher">
            <button class="lang-switcher-btn" id="langSwitcherBtn" aria-expanded="false" aria-haspopup="true">
                <span class="lang-flag">${currentLang?.flag || '🌐'}</span>
                <span class="lang-name">${currentLang?.code?.toUpperCase() || 'EN'}</span>
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <path d="M6 9l6 6 6-6"/>
                </svg>
            </button>
            <div class="lang-dropdown" id="langDropdown" style="display: none;">
                ${SUPPORTED_LANGUAGES.map(lang => `
                    <button class="lang-option ${lang.code === currentLanguage ? 'active' : ''}" data-lang="${lang.code}">
                        <span class="lang-flag">${lang.flag}</span>
                        <span class="lang-native">${lang.nativeName}</span>
                        <span class="lang-english">${lang.name}</span>
                    </button>
                `).join('')}
            </div>
        </div>
    `;

    // Add event listeners
    const btn = container.querySelector('#langSwitcherBtn');
    const dropdown = container.querySelector('#langDropdown');

    btn?.addEventListener('click', () => {
        const isOpen = dropdown.style.display !== 'none';
        dropdown.style.display = isOpen ? 'none' : 'block';
        btn.setAttribute('aria-expanded', !isOpen);
    });

    container.querySelectorAll('.lang-option').forEach(option => {
        option.addEventListener('click', async () => {
            const lang = option.dataset.lang;
            await setLanguage(lang);
            dropdown.style.display = 'none';
            btn.setAttribute('aria-expanded', 'false');

            // Update button text
            const selected = SUPPORTED_LANGUAGES.find(l => l.code === lang);
            btn.querySelector('.lang-flag').textContent = selected?.flag || '🌐';
            btn.querySelector('.lang-name').textContent = selected?.code?.toUpperCase() || 'EN';

            // Update active state
            container.querySelectorAll('.lang-option').forEach(o => o.classList.remove('active'));
            option.classList.add('active');
        });
    });

    // Close dropdown when clicking outside
    document.addEventListener('click', (e) => {
        if (!container.contains(e.target)) {
            dropdown.style.display = 'none';
            btn?.setAttribute('aria-expanded', 'false');
        }
    });
}

// ── Initialization ─────────────────────────────────────────────────────

export async function initI18n() {
    if (isInitialized) return;

    // Detect language
    currentLanguage = detectLanguage();

    // Load translations for detected language
    await loadTranslations(currentLanguage);

    // Also load default language as fallback
    if (currentLanguage !== DEFAULT_LANGUAGE) {
        await loadTranslations(DEFAULT_LANGUAGE);
    }

    // Set HTML lang attribute
    document.documentElement.lang = currentLanguage;

    // Update translatable elements
    updateTranslatableElements();

    // Create language switcher if container exists
    createLanguageSwitcher();

    isInitialized = true;
    // console.log removed for production
}

