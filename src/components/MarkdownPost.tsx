import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'

type MarkdownPostProps = {
    content: string
}

export function MarkdownPost({ content }: MarkdownPostProps) {
    return (
        <div className="markdown-body">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </div>
    )
}
