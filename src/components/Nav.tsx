import { NavLink } from 'react-router-dom'

export function Nav() {
    return (
        <header className="site-header">
            <nav className="site-nav" aria-label="Primary navigation">
                <NavLink to="/" className="site-mark">
                    Huang Xinying
                </NavLink>

                <div className="nav-links">
                    <NavLink to="/" end>
                        Home
                    </NavLink>
                    <NavLink to="/blog">Blog</NavLink>
                    <a href="https://github.com/crashingby">GitHub</a>
                </div>
            </nav>
        </header>
    )
}
