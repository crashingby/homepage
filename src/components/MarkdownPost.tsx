import {
    isValidElement,
    useEffect,
    useId,
    useState,
    type ComponentPropsWithoutRef,
    type ReactElement,
    type ReactNode,
} from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

type MarkdownPostProps = {
    content: string
}

type CodeElementProps = {
    className?: string
    children?: ReactNode
}

function MermaidDiagram({ chart }: { chart: string }) {
    const rawId = useId()
    const id = `mermaid-${rawId.replace(/[^a-zA-Z0-9-_]/g, '')}`
    const [svg, setSvg] = useState('')
    const [error, setError] = useState('')

    useEffect(() => {
        let cancelled = false

        async function renderDiagram() {
            try {
                const { default: mermaid } = await import('mermaid')

                mermaid.initialize({
                    startOnLoad: false,
                    securityLevel: 'strict',
                    theme: 'dark',
                })

                const result = await mermaid.render(id, chart)

                if (!cancelled) {
                    setSvg(result.svg)
                    setError('')
                }
            } catch (err) {
                if (!cancelled) {
                    setSvg('')
                    setError(err instanceof Error ? err.message : 'Unable to render Mermaid diagram.')
                }
            }
        }

        void renderDiagram()

        return () => {
            cancelled = true
        }
    }, [chart, id])

    if (error) {
        return (
            <div className="mermaid-diagram mermaid-error">
                <p>Unable to render Mermaid diagram.</p>
                <code>{chart}</code>
            </div>
        )
    }

    if (!svg) {
        return <div className="mermaid-diagram mermaid-loading">Rendering diagram...</div>
    }

    return (
        <div
            className="mermaid-diagram"
            dangerouslySetInnerHTML={{ __html: svg }}
            role="img"
        />
    )
}

function PreBlock({ children, ...props }: ComponentPropsWithoutRef<'pre'>) {
    if (isValidElement(children)) {
        const code = children as ReactElement<CodeElementProps>
        const language = code.props.className?.replace('language-', '')

        if (language === 'mermaid') {
            return <MermaidDiagram chart={String(code.props.children).replace(/\n$/, '')} />
        }
    }

    return <pre {...props}>{children}</pre>
}

function CodeBlock({ className, children, ...props }: ComponentPropsWithoutRef<'code'>) {
    return (
        <code className={className} {...props}>
            {children}
        </code>
    )
}

function TableBlock({ children, ...props }: ComponentPropsWithoutRef<'table'>) {
    return (
        <div className="table-scroll">
            <table {...props}>{children}</table>
        </div>
    )
}

export function MarkdownPost({ content }: MarkdownPostProps) {
    return (
        <div className="markdown-body">
            <ReactMarkdown
                components={{
                    code: CodeBlock,
                    pre: PreBlock,
                    table: TableBlock,
                }}
                remarkPlugins={[remarkGfm]}
            >
                {content}
            </ReactMarkdown>
        </div>
    )
}
