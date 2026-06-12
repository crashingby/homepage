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
import { getMarkdownHeadings } from '../lib/markdown'

type MarkdownPostProps = {
    content: string
}

type CodeElementProps = {
    className?: string
    children?: ReactNode
}

type MarkdownAstNode = {
    position?: {
        start?: {
            line?: number
        }
    }
}

type HeadingProps<Level extends 1 | 2 | 3 | 4 | 5 | 6> = ComponentPropsWithoutRef<
    `h${Level}`
> & {
    node?: MarkdownAstNode
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

function getNodeText(children: ReactNode): string {
    if (typeof children === 'string' || typeof children === 'number') {
        return String(children)
    }

    if (Array.isArray(children)) {
        return children.map(getNodeText).join('')
    }

    if (isValidElement(children)) {
        return getNodeText((children as ReactElement<{ children?: ReactNode }>).props.children)
    }

    return ''
}

function createHeading(level: 1 | 2 | 3 | 4 | 5 | 6, headingIdsByLine: Map<number, string>) {
    const Heading = `h${level}` as const

    return function HeadingBlock({ children, node, ...props }: HeadingProps<typeof level>) {
        const line = node?.position?.start?.line
        const id = line ? headingIdsByLine.get(line) : undefined

        return (
            <Heading id={id ?? getNodeText(children)} {...props}>
                {children}
            </Heading>
        )
    }
}

export function MarkdownPost({ content }: MarkdownPostProps) {
    const headingIdsByLine = new Map(
        getMarkdownHeadings(content).map((heading) => [heading.line, heading.id]),
    )

    return (
        <div className="markdown-body">
            <ReactMarkdown
                components={{
                    code: CodeBlock,
                    h1: createHeading(1, headingIdsByLine),
                    h2: createHeading(2, headingIdsByLine),
                    h3: createHeading(3, headingIdsByLine),
                    h4: createHeading(4, headingIdsByLine),
                    h5: createHeading(5, headingIdsByLine),
                    h6: createHeading(6, headingIdsByLine),
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
