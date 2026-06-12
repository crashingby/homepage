const projects = [
    {
        title: 'GEMM 迭代优化（进行中）',
        desc: '逐版本迭代GEMM。使用主流的优化路径：共享内存，异步拷贝，Tensor Core。',
        href: 'https://gitee.com/hxy21211319/gemm-reseach',
    },
    {
        title: 'Reduce 迭代优化（已完成）',
        desc: '逐版本迭代Reduce算子。已达到显卡最大吞吐90%以上。提供CUTE和Triton版本的最终版。',
        href: 'https://gitee.com/hxy21211319/reduce-research',
    },
    {
        title: 'AI推理服务系统（进行中）',
        desc: 'CPP后端+AI部署的结合。正在进行中。',
        href: 'https://gitee.com/hxy21211319/InferenceServers',
    },
]

export function ProjectPreview() {
    return (
        <section className="section" id="projects">
            <h2>Projects</h2>

            <div className="project-grid">
                {projects.map((project) => (
                    <article className="project-card" key={project.title}>
                        <h3>{project.title}</h3>
                        <p>{project.desc}</p>
                        <a
                            className="project-link"
                            href={project.href}
                            target="_blank"
                            rel="noreferrer"
                        >
                            查看 Gitee
                        </a>
                    </article>
                ))}
            </div>
        </section>
    )
}
