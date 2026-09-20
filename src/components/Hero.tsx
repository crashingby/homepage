import { Link } from 'react-router-dom'
import { getBlogPosts, getBlogTopics } from '../lib/blog'
import { Icon } from './Icon'

export function Hero() {
    return (
        <section className="hero">
            <div className="hero-copy">
                <p className="eyebrow">
                    <span className="status-dot" /> A PERSONAL FIELD NOTEBOOK
                </p>
                <h1>
                    Crashing By<span>.</span>
                </h1>
                <p className="hero-heading">
                    在代码与系统之间，
                    <br />
                    留下一些思考。
                </p>
                <p className="hero-subtitle">
                    我是 Huang Xinying，硕士在读。
                    <br />写 C++，研究 CUDA 与 GPU，也记录一路上的发现。
                </p>
                <div className="hero-actions">
                    <Link to="/blog" className="primary-button">
                        翻开笔记 <Icon name="arrow" />
                    </Link>
                    <a
                        href="#projects"
                        className="text-link"
                        onClick={(event) => {
                            event.preventDefault()
                            document
                                .getElementById('projects')
                                ?.scrollIntoView({ behavior: 'smooth' })
                        }}
                    >
                        看看我的项目 <Icon name="upRight" size={16} />
                    </a>
                </div>
            </div>
            <aside className="field-notes" aria-label="笔记概览">
                <div className="field-notes-top">
                    <span>THE NOTEBOOK</span>
                    <span>01 / ∞</span>
                </div>
                <div className="notebook-diagram" aria-hidden="true">
                    <span>THINK</span>
                    <i />
                    <span>BUILD</span>
                    <i />
                    <span>UNDERSTAND</span>
                    <div className="diagram-caption">
                        observe → question → iterate
                    </div>
                </div>
                <p>
                    从一行代码，
                    <br />
                    到系统的全貌。
                </p>
                <div className="notebook-topics">
                    <span>GPU Programming</span>
                    <span>C++ & Systems</span>
                    <span>AI Infrastructure</span>
                </div>
                <div className="notebook-stats">
                    <span>
                        <strong>{getBlogPosts().length}</strong> 篇笔记
                    </span>
                    <span>
                        <strong>{getBlogTopics().length}</strong> 个专题
                    </span>
                    <span className="status-dot" />
                </div>
            </aside>
        </section>
    )
}
