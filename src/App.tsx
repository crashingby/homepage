import './App.css'
import { HashRouter, Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { BlogIndexPage } from './pages/BlogIndexPage'
import { BlogPostPage } from './pages/BlogPostPage'
import { HomePage } from './pages/HomePage'

function App() {
    return (
        <HashRouter>
            <Routes>
                <Route element={<Layout />}>
                    <Route index element={<HomePage />} />
                    <Route path="blog" element={<BlogIndexPage />} />
                    <Route path="blog/topic/:topicSlug" element={<BlogIndexPage />} />
                    <Route path="blog/:topicSlug/:slug" element={<BlogPostPage />} />
                    <Route path="blog/:slug" element={<BlogPostPage />} />
                    <Route path="*" element={<Navigate to="/" replace />} />
                </Route>
            </Routes>
        </HashRouter>
    )
}

export default App
