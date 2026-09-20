import {
    isValidElement,
    memo,
    useEffect,
    useId,
    useState,
    type ComponentPropsWithoutRef,
    type ReactElement,
    type ReactNode,
} from 'react'
import { Link } from 'react-router-dom'
import rehypeKatex from 'rehype-katex'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import remarkMath from 'remark-math'
import 'katex/dist/katex.min.css'
import { getMarkdownHeadings } from '../lib/markdown'
import { Icon } from './Icon'

type MarkdownPostProps = {
    content: string
    hiddenHeadingId?: string
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

type HeadingProps<Level extends 1 | 2 | 3 | 4 | 5 | 6> =
    ComponentPropsWithoutRef<`h${Level}`> & {
        node?: MarkdownAstNode
    }

const languageLabels: Record<string, string> = {
    bash: 'Bash',
    cpp: 'C++',
    cuda: 'CUDA',
    cmake: 'CMake',
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
    cuda: 'cpp',
    cu: 'cpp',
    cxx: 'cpp',
    cmake: 'cmake',
    yaml: 'yaml',
    yml: 'yaml',
    toml: 'toml',
    rust: 'rust',
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
    codeToHtml: (
        code: string,
        options: { lang: string; theme: string },
    ) => string
    loadLanguage: unknown
}

let shikiHighlighterPromise: Promise<ShikiHighlighter> | null = null
const loadedShikiLanguages = new Set<string>()

const shikiLanguageLoaders: Record<
    string,
    () => Promise<{ default: unknown }>
> = {
    bash: () => import('@shikijs/langs/bash'),
    cpp: () => import('@shikijs/langs/cpp'),
    cmake: () => import('@shikijs/langs/cmake'),
    yaml: () => import('@shikijs/langs/yaml'),
    toml: () => import('@shikijs/langs/toml'),
    rust: () => import('@shikijs/langs/rust'),
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

async function loadShikiLanguage(
    highlighter: ShikiHighlighter,
    language: string,
) {
    if (loadedShikiLanguages.has(language)) {
        return
    }

    const loader = shikiLanguageLoaders[language]

    if (!loader) {
        return
    }

    const { default: grammar } = await loader()
    await (highlighter.loadLanguage as (language: unknown) => Promise<void>)(
        grammar,
    )
    loadedShikiLanguages.add(language)
}

function CodeFrame({
    code,
    language,
    children,
}: {
    code: string
    language?: string
    children: ReactNode
}) {
    const [copied, setCopied] = useState(false)
    const [copyError, setCopyError] = useState(false)
    const [wrapped, setWrapped] = useState(false)
    useEffect(() => {
        if (!copied && !copyError) return
        const timer = window.setTimeout(() => {
            setCopied(false)
            setCopyError(false)
        }, 2000)
        return () => window.clearTimeout(timer)
    }, [copied, copyError])
    const copy = async () => {
        try {
            await navigator.clipboard.writeText(code)
            setCopied(true)
        } catch {
            setCopyError(true)
        }
    }
    return (
        <figure className={`code-block${wrapped ? ' is-wrapped' : ''}`}>
            <figcaption className="code-block-header">
                <span>{getLanguageLabel(language)}</span>
                <span className="code-block-actions">
                    <button
                        type="button"
                        aria-pressed={wrapped}
                        onClick={() => setWrapped(!wrapped)}
                    >
                        自动换行
                    </button>
                    <button
                        type="button"
                        onClick={() => void copy()}
                        aria-label="复制代码"
                    >
                        <Icon name={copied ? 'check' : 'copy'} size={12} />
                        <span role="status">
                            {copied
                                ? '已复制'
                                : copyError
                                  ? '请手动复制'
                                  : '复制'}
                        </span>
                    </button>
                </span>
            </figcaption>
            {children}
        </figure>
    )
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
        <CodeFrame code={code} language={language}>
            <pre {...preProps} tabIndex={0}>
                <code>{code}</code>
            </pre>
        </CodeFrame>
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
        return (
            <PlainCodeBlock
                code={code}
                language={language}
                preProps={preProps}
            />
        )
    }

    return (
        <CodeFrame code={code} language={language}>
            <div
                className="shiki-code"
                dangerouslySetInnerHTML={{ __html: html }}
            />
        </CodeFrame>
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
                        primaryColor: '#e8eee5',
                        primaryTextColor: '#252923',
                        primaryBorderColor: '#77916e',
                        lineColor: '#656b61',
                        secondaryColor: '#f1eade',
                        tertiaryColor: '#f8f7f4',
                        fontFamily:
                            'Inter, ui-sans-serif, system-ui, sans-serif',
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
                    setError(
                        err instanceof Error
                            ? err.message
                            : 'Unable to render Mermaid diagram.',
                    )
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
        return (
            <div className="mermaid-diagram mermaid-loading">
                Rendering diagram...
            </div>
        )
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

        return (
            <ShikiCodeBlock
                code={codeText}
                language={language}
                preProps={props}
            />
        )
    }

    return <PlainCodeBlock code={String(children)} preProps={props} />
}

function CodeBlock({
    className,
    children,
    ...props
}: ComponentPropsWithoutRef<'code'>) {
    return (
        <code className={className} {...props}>
            {children}
        </code>
    )
}

function TableBlock({ children, ...props }: ComponentPropsWithoutRef<'table'>) {
    return (
        <div className="table-scroll" tabIndex={0} role="region" aria-label="文章表格，可横向滚动">
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
        return getNodeText(
            (children as ReactElement<{ children?: ReactNode }>).props.children,
        )
    }

    return ''
}

function createHeading(
    level: 1 | 2 | 3 | 4 | 5 | 6,
    headingIdsByLine: Map<number, string>,
    hiddenHeadingId?: string,
) {
    const Heading = `h${level === 1 ? 2 : level}` as const

    return function HeadingBlock({
        children,
        node,
        ...props
    }: HeadingProps<typeof level>) {
        const line = node?.position?.start?.line
        const id = line ? headingIdsByLine.get(line) : undefined

        if (id && id === hiddenHeadingId) return null

        return (
            <Heading id={id ?? getNodeText(children)} {...props}>
                {children}
                {id && (
                    <Link
                        to={{ hash: `#${id}` }}
                        className="heading-anchor"
                        aria-label={`跳转到 ${getNodeText(children)}`}
                    >
                        #
                    </Link>
                )}
            </Heading>
        )
    }
}

function MarkdownLink({
    href,
    children,
    ...props
}: ComponentPropsWithoutRef<'a'>) {
    if (href?.startsWith('#/'))
        return <Link to={href.slice(1)}>{children}</Link>
    if (href?.startsWith('#'))
        return <Link to={{ hash: href }}>{children}</Link>
    return (
        <a href={href} {...props}>
            {children}
        </a>
    )
}

export const MarkdownPost = memo(function MarkdownPost({
    content,
    hiddenHeadingId,
}: MarkdownPostProps) {
    const headingIdsByLine = new Map(
        getMarkdownHeadings(content).map((heading) => [
            heading.line,
            heading.id,
        ]),
    )

    return (
        <div className="markdown-body">
            <ReactMarkdown
                components={{
                    a: MarkdownLink,
                    code: CodeBlock,
                    h1: createHeading(1, headingIdsByLine, hiddenHeadingId),
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
})
