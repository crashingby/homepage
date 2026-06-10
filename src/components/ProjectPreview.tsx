const projects = [
    {
        title: 'GEMM Optimization',
        desc: 'Iterative CUDA GEMM optimization from naive kernels to shared memory tiling, register tiling, CuTe, Triton, and Tensor Core paths.',
    },
    {
        title: 'Reduce Research',
        desc: 'CUDA, CuTe, Triton, and TileLang implementations for reduction kernels with performance-oriented benchmarking.',
    },
    {
        title: 'BFS Fault Tolerance',
        desc: 'GPU BFS with DMR, top-k critical vertex selection, threshold-based redundancy, and multi-GPU NCCL exploration.',
    },
    {
        title: 'LLM Inference Systems',
        desc: 'Exploration of incremental prefill, KV cache scheduling, streaming RAG, and inference serving architecture.',
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
                    </article>
                ))}
            </div>
        </section>
    )
}