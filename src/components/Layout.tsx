import { Outlet } from 'react-router-dom'
import { Nav } from './Nav'

export function Layout() {
    return (
        <>
            <Nav />
            <main className="app">
                <Outlet />
            </main>
        </>
    )
}
