import { useEffect } from 'react'
import { Link } from 'react-router-dom'
import { About } from '../components/About'
import { Hero } from '../components/Hero'
import { Icon } from '../components/Icon'
import { ProjectPreview } from '../components/ProjectPreview'
import { PostList } from '../components/PostList'
import { getBlogPosts } from '../lib/blog'

export function HomePage() {
    useEffect(() => {
        document.title = 'Crashing By — 技术与日常'
    }, [])
    return (
        <div className="home-page">
            <Hero />
            <section className="section recent-notes">
                <div className="section-heading">
                    <div>
                        <p className="eyebrow">RECENT WRITING</p>
                        <h2>
                            最近的笔记<span>持续记录，慢慢理解。</span>
                        </h2>
                    </div>
                    <Link to="/blog" className="text-link">
                        全部笔记 <Icon name="arrow" size={16} />
                    </Link>
                </div>
                <PostList posts={getBlogPosts().slice(0, 4)} numbered />
            </section>
            <ProjectPreview />
            <About />
        </div>
    )
}
