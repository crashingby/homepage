import { useEffect } from 'react'
import { Link, Outlet, useLocation } from 'react-router-dom'
import { Nav } from './Nav'
import { MusicPlayer } from './MusicPlayer'

export function Layout() {
    const { pathname } = useLocation()
    useEffect(() => {
        window.scrollTo({ top: 0, behavior: 'instant' })
    }, [pathname])
    return (
        <>
            <a
                className="skip-link"
                href="#main-content"
                onClick={(event) => {
                    event.preventDefault()
                    document.getElementById('main-content')?.focus()
                }}
            >
                跳至正文
            </a>
            <Nav />
            <main className="app" id="main-content" tabIndex={-1}>
                <Outlet />
            </main>
            <footer className="site-footer">
                <Link to="/">
                    Crashing By<span>记录、理解，再向前一点。</span>
                </Link>
                <span>GPU · SYSTEMS · LIFE</span>
            </footer>
            <MusicPlayer />
        </>
    )
}
