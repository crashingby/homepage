import { useEffect, useState } from 'react'
import type { MarkdownHeading } from './markdown'

export function useReadingProgress(headings: MarkdownHeading[]) {
    const [state, setState] = useState({ activeId: '', progress: 0 })
    useEffect(() => {
        const article = document.querySelector<HTMLElement>('.article-content')
        if (!article) return
        let frame = 0
        const update = () => {
            frame = 0
            const elements = headings
                .map((heading) => document.getElementById(heading.id))
                .filter((element): element is HTMLElement => element !== null)
            let activeId = elements[0]?.id ?? ''
            for (const element of elements) {
                if (element.getBoundingClientRect().top <= 160)
                    activeId = element.id
                else break
            }
            const bounds = article.getBoundingClientRect()
            const distance = bounds.height - window.innerHeight + 120
            const rawProgress = Math.max(
                0,
                Math.min(
                    100,
                    distance > 0
                        ? ((120 - bounds.top) / distance) * 100
                        : bounds.bottom <= window.innerHeight
                          ? 100
                          : 0,
                ),
            )
            const progress = Math.round(rawProgress)
            setState((current) =>
                current.activeId === activeId &&
                current.progress === progress
                    ? current
                    : { activeId, progress },
            )
        }
        const schedule = () => {
            if (!frame) frame = requestAnimationFrame(update)
        }
        const resizeObserver = new ResizeObserver(schedule)
        resizeObserver.observe(article)
        window.addEventListener('scroll', schedule, { passive: true })
        window.addEventListener('resize', schedule)
        schedule()
        return () => {
            cancelAnimationFrame(frame)
            resizeObserver.disconnect()
            window.removeEventListener('scroll', schedule)
            window.removeEventListener('resize', schedule)
        }
    }, [headings])
    return state
}
