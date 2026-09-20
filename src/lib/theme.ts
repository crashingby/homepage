import { useSyncExternalStore } from 'react'

type Theme = 'light' | 'dark'
function getTheme(): Theme {
    return document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'
}
function subscribe(callback: () => void) {
    window.addEventListener('themechange', callback)
    return () => window.removeEventListener('themechange', callback)
}
export function useTheme() {
    const theme = useSyncExternalStore(
        subscribe,
        getTheme,
        () => 'light' as const,
    )
    const toggleTheme = () => {
        const next = getTheme() === 'light' ? 'dark' : 'light'
        document.documentElement.dataset.theme = next
        try {
            localStorage.setItem('site-theme', next)
        } catch {
            /* Storage can be unavailable. */
        }
        window.dispatchEvent(new Event('themechange'))
    }
    return { theme, toggleTheme }
}
