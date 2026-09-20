import { NavLink } from 'react-router-dom'
import { useTheme } from '../lib/theme'
import { Icon } from './Icon'

export function Nav() {
    const { theme, toggleTheme } = useTheme()
    return (
        <header className="site-header">
            <nav className="site-nav" aria-label="主导航">
                <NavLink
                    to="/"
                    className="site-mark"
                    aria-label="Crashing By 首页"
                >
                    <span className="site-monogram">cb.</span>
                    <span>
                        Crashing By
                        <span className="site-mark-caption">Huang Xinying</span>
                    </span>
                </NavLink>
                <div className="nav-links">
                    <NavLink to="/" end>
                        首页
                    </NavLink>
                    <NavLink to="/blog">笔记</NavLink>
                    <a
                        href="https://github.com/crashingby"
                        target="_blank"
                        rel="noreferrer"
                        className="nav-github"
                    >
                        GitHub <Icon name="upRight" size={14} />
                    </a>
                    <span className="nav-divider" />
                    <button
                        type="button"
                        className="icon-button"
                        onClick={toggleTheme}
                        aria-label={
                            theme === 'light' ? '切换深色模式' : '切换浅色模式'
                        }
                        title={
                            theme === 'light' ? '切换深色模式' : '切换浅色模式'
                        }
                    >
                        <Icon name={theme === 'light' ? 'moon' : 'sun'} />
                    </button>
                </div>
            </nav>
        </header>
    )
}
