import { Outlet } from 'react-router-dom'
import { Nav } from './Nav'
import { ParticleBackground } from './ParticleBackground'

export function Layout() {
    return (
        <>
            <ParticleBackground />
            <Nav />
            <main className="app">
                <Outlet />
            </main>
        </>
    )
}
