import './styles/global.css'
import './styles/layout.css'
import './styles/home.css'
import './styles/blog.css'
import './styles/markdown.css'
import './styles/music.css'
import './styles/interview.css'
import { HashRouter, Navigate, Route, Routes } from 'react-router-dom'
import { Layout } from './components/Layout'
import { BlogIndexPage } from './pages/BlogIndexPage'
import { BlogPostPage } from './pages/BlogPostPage'
import { HomePage } from './pages/HomePage'
import { InterviewLayout } from './layouts/InterviewLayout'
import { InterviewIndexPage } from './pages/interview/InterviewIndexPage'
import {
    InterviewChapterPage,
    InterviewNotFoundPage,
} from './pages/interview/InterviewChapterPage'
import { LegacyInterviewRedirect } from './pages/interview/LegacyInterviewRedirect'

function App() {
    return (
        <HashRouter>
            <Routes>
                <Route path="blog/topic/interview-guide" element={<LegacyInterviewRedirect />} />
                <Route path="blog/interview-guide/:slug" element={<LegacyInterviewRedirect />} />
                <Route path="interview" element={<InterviewLayout />}>
                    <Route index element={<InterviewIndexPage />} />
                    <Route path=":trackSlug/:slug" element={<InterviewChapterPage />} />
                    <Route path="*" element={<InterviewNotFoundPage />} />
                </Route>
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
