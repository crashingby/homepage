import { Link } from 'react-router-dom'
import { getBlogCategoryPath, getBlogSeriesPosts, type BlogPost } from '../lib/blog'
import { Icon } from './Icon'

export function BlogSeriesNav({ post }: { post: BlogPost }) {
    const posts = getBlogSeriesPosts(post)
    const index = posts.findIndex((item) => item.path === post.path)
    if (!post.series || index < 0 || posts.length < 2) return null

    const previous = posts[index - 1]
    const next = posts[index + 1]

    return (
        <nav className="article-series" aria-label="系列文章导航">
            <div className="series-header">
                <div>
                    <p className="eyebrow">系列阅读 · 第 {index + 1} / {posts.length} 篇</p>
                    <h2>{post.series.label}</h2>
                </div>
                <Link to={getBlogCategoryPath(post.topic.slug, post.category?.slug)} className="text-link">
                    全部章节 <Icon name="list" size={15} />
                </Link>
            </div>
            <div className="series-pagination">
                {previous && (
                    <Link to={previous.path} rel="prev" className="series-previous">
                        <span><Icon name="back" size={15} /> 上一篇</span>
                        <strong>{previous.title}</strong>
                    </Link>
                )}
                {next && (
                    <Link to={next.path} rel="next" className="series-next">
                        <span>下一篇 <Icon name="arrow" size={15} /></span>
                        <strong>{next.title}</strong>
                    </Link>
                )}
            </div>
        </nav>
    )
}
