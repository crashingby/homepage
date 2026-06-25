import {
    isValidElement,
    useEffect,
    useId,
    useState,
    type ComponentPropsWithoutRef,
    type ReactElement,
    type ReactNode,
} from 'react'
import rehypeKatex from 'rehype-katex'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import 'katex/dist/katex.min.css'
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

const languageLabels: Record<string, string> = {
    bash: 'Bash',
    cpp: 'C++',
    'c++': 'C++',
    javascript: 'JavaScript',
    js: 'JavaScript',
    json: 'JSON',
    python: 'Python',
    sh: 'Shell',
    shell: 'Shell',
    ts: 'TypeScript',
    typescript: 'TypeScript',
}

const shikiLanguageAliases: Record<string, string> = {
    bash: 'bash',
    c: 'cpp',
    cpp: 'cpp',
    'c++': 'cpp',
    javascript: 'javascript',
    js: 'javascript',
    json: 'json',
    python: 'python',
    py: 'python',
    sh: 'shellscript',
    shell: 'shellscript',
    shellscript: 'shellscript',
    ts: 'typescript',
    typescript: 'typescript',
}

type ShikiHighlighter = {
    codeToHtml: (code: string, options: { lang: string; theme: string }) => string
    loadLanguage: unknown
}

let shikiHighlighterPromise: Promise<ShikiHighlighter> | null = null
const loadedShikiLanguages = new Set<string>()

const shikiLanguageLoaders: Record<string, () => Promise<{ default: unknown }>> = {
    bash: () => import('@shikijs/langs/bash'),
    cpp: () => import('@shikijs/langs/cpp'),
    javascript: () => import('@shikijs/langs/javascript'),
    json: () => import('@shikijs/langs/json'),
    python: () => import('@shikijs/langs/python'),
    shellscript: () => import('@shikijs/langs/shellscript'),
    typescript: () => import('@shikijs/langs/typescript'),
}

function getLanguageLabel(language?: string) {
    if (!language) {
        return 'Code'
    }

    return languageLabels[language] ?? language.toUpperCase()
}

function getShikiLanguage(language?: string) {
    if (!language) {
        return undefined
    }

    return shikiLanguageAliases[language.toLowerCase()]
}

async function getShikiHighlighter() {
    shikiHighlighterPromise ??= Promise.all([
        import('shiki/core'),
        import('shiki/engine/javascript'),
        import('@shikijs/themes/dark-plus'),
    ]).then(
        ([
            { createHighlighterCore },
            { createJavaScriptRegexEngine },
            { default: darkPlus },
        ]) =>
            createHighlighterCore({
                engine: createJavaScriptRegexEngine(),
                langs: [],
                themes: [darkPlus],
            }).then((highlighter) => highlighter as ShikiHighlighter),
    )

    const highlighter = await shikiHighlighterPromise
    return highlighter
}

async function loadShikiLanguage(highlighter: ShikiHighlighter, language: string) {
    if (loadedShikiLanguages.has(language)) {
        return
    }

    const loader = shikiLanguageLoaders[language]

    if (!loader) {
        return
    }

    const { default: grammar } = await loader()
    await (highlighter.loadLanguage as (language: unknown) => Promise<void>)(grammar)
    loadedShikiLanguages.add(language)
}

function PlainCodeBlock({
    code,
    language,
    preProps,
}: {
    code: string
    language?: string
    preProps?: ComponentPropsWithoutRef<'pre'>
}) {
    return (
        <figure className="code-block">
            <figcaption className="code-block-header">
                <span>{getLanguageLabel(language)}</span>
            </figcaption>
            <pre {...preProps} tabIndex={0}>
                <code>{code}</code>
            </pre>
        </figure>
    )
}

function ShikiCodeBlock({
    code,
    language,
    preProps,
}: {
    code: string
    language?: string
    preProps?: ComponentPropsWithoutRef<'pre'>
}) {
    const [html, setHtml] = useState('')
    const shikiLanguage = getShikiLanguage(language)

    useEffect(() => {
        let cancelled = false

        async function highlightCode() {
            if (!shikiLanguage) {
                setHtml('')
                return
            }

            try {
                const highlighter = await getShikiHighlighter()
                await loadShikiLanguage(highlighter, shikiLanguage)
                const highlighted = highlighter.codeToHtml(code, {
                    lang: shikiLanguage,
                    theme: 'dark-plus',
                })

                if (!cancelled) {
                    setHtml(highlighted)
                }
            } catch {
                if (!cancelled) {
                    setHtml('')
                }
            }
        }

        void highlightCode()

        return () => {
            cancelled = true
        }
    }, [code, shikiLanguage])

    if (!html) {
        return <PlainCodeBlock code={code} language={language} preProps={preProps} />
    }

    return (
        <figure className="code-block">
            <figcaption className="code-block-header">
                <span>{getLanguageLabel(language)}</span>
            </figcaption>
            <div className="shiki-code" dangerouslySetInnerHTML={{ __html: html }} />
        </figure>
    )
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
                    theme: 'base',
                    themeVariables: {
                        background: '#ffffff',
                        primaryColor: '#dbeafe',
                        primaryTextColor: '#0f172a',
                        primaryBorderColor: '#60a5fa',
                        lineColor: '#64748b',
                        secondaryColor: '#fce7f3',
                        tertiaryColor: '#f8fafc',
                        fontFamily: 'Inter, ui-sans-serif, system-ui, sans-serif',
                    },
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
        const codeText = String(code.props.children).replace(/\n$/, '')

        if (language === 'mermaid') {
            return <MermaidDiagram chart={codeText} />
        }

        return <ShikiCodeBlock code={codeText} language={language} preProps={props} />
    }

    return <PlainCodeBlock code={String(children)} preProps={props} />
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
                rehypePlugins={[rehypeKatex]}
                remarkPlugins={[remarkGfm, remarkMath]}
            >
                {content}
            </ReactMarkdown>
        </div>
    )
}
