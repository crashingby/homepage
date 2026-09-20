import { Icon } from './Icon'

const projects = [
    {
        title: 'GEMM 迭代优化',
        status: '已完成',
        desc: '沿着共享内存、异步拷贝与 Tensor Core 的路径，逐版本探索矩阵乘的性能边界。',
        tags: 'CUDA / Tensor Core',
        href: 'https://gitee.com/hxy21211319/gemm-reseach',
    },
    {
        title: 'Reduce 迭代优化',
        status: '已完成',
        desc: '逐步优化 Reduce 算子，达到显卡最大吞吐的 90% 以上，提供 CuTe 与 Triton 实现。',
        tags: 'CuTe / Triton',
        href: 'https://gitee.com/hxy21211319/reduce-research',
    },
    {
        title: 'AI 推理服务系统',
        status: '进行中',
        desc: '把 C++ 后端与 AI 部署连接起来，探索推理服务的工程实现。',
        tags: 'C++ / AI Serving',
        href: 'https://gitee.com/hxy21211319/InferenceServers',
    },
]
export function ProjectPreview() {
    return (
        <section className="section" id="projects">
            <div className="section-heading">
                <div>
                    <p className="eyebrow">LEARNING BY BUILDING</p>
                    <h2>动手做的事</h2>
                </div>
                <span className="section-caption">从理解到实现</span>
            </div>
            <div className="project-grid">
                {projects.map((project, index) => (
                    <a
                        className="project-card"
                        key={project.title}
                        href={project.href}
                        target="_blank"
                        rel="noreferrer"
                    >
                        <div className="project-top">
                            <span className="mono">0{index + 1}</span>
                            <span
                                className={`project-status${project.status === '进行中' ? ' is-ongoing' : ''}`}
                            >
                                {project.status}
                            </span>
                        </div>
                        <h3>
                            {project.title} <Icon name="upRight" size={18} />
                        </h3>
                        <p>{project.desc}</p>
                        <span className="project-stack">{project.tags}</span>
                    </a>
                ))}
            </div>
        </section>
    )
}
