import { useEffect, useRef, useState } from 'react'
import { musicTracks } from '../data/music'
import { Icon } from './Icon'

function formatTime(seconds: number) {
    if (!Number.isFinite(seconds)) return '0:00'
    return `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`
}

export function MusicPlayer() {
    const audioRef = useRef<HTMLAudioElement>(null)
    const containerRef = useRef<HTMLDivElement>(null)
    const triggerRef = useRef<HTMLButtonElement>(null)
    const wantsPlayback = useRef(false)
    const [trackIndex, setTrackIndex] = useState(0)
    const [isPlaying, setIsPlaying] = useState(false)
    const [isOpen, setIsOpen] = useState(false)
    const [volume, setVolume] = useState(0.72)
    const [currentTime, setCurrentTime] = useState(0)
    const [duration, setDuration] = useState(0)
    const [error, setError] = useState('')
    const currentTrack = musicTracks[trackIndex]

    useEffect(() => {
        const audio = audioRef.current
        if (!audio || !currentTrack) return
        let cancelled = false
        audio.load()
        if (wantsPlayback.current)
            void audio.play().catch((reason: unknown) => {
                if (
                    !cancelled &&
                    !(
                        reason instanceof DOMException &&
                        reason.name === 'AbortError'
                    )
                ) {
                    wantsPlayback.current = false
                    setError('暂时无法播放，请检查网络后重试。')
                }
            })
        return () => {
            cancelled = true
        }
    }, [currentTrack])

    useEffect(() => {
        if (audioRef.current) audioRef.current.volume = volume
    }, [volume])
    useEffect(() => {
        if (!isOpen) return
        const close = (event: MouseEvent) => {
            if (!containerRef.current?.contains(event.target as Node))
                setIsOpen(false)
        }
        const escape = (event: KeyboardEvent) => {
            if (event.key === 'Escape') {
                setIsOpen(false)
                triggerRef.current?.focus()
            }
        }
        document.addEventListener('mousedown', close)
        document.addEventListener('keydown', escape)
        return () => {
            document.removeEventListener('mousedown', close)
            document.removeEventListener('keydown', escape)
        }
    }, [isOpen])

    if (!currentTrack) return null
    const goToTrack = (index: number) => {
        setError('')
        setCurrentTime(0)
        setDuration(0)
        setTrackIndex((index + musicTracks.length) % musicTracks.length)
    }
    const togglePlayback = async () => {
        const audio = audioRef.current
        if (!audio) return
        setError('')
        if (!audio.paused) {
            wantsPlayback.current = false
            audio.pause()
            return
        }
        wantsPlayback.current = true
        try {
            await audio.play()
        } catch {
            wantsPlayback.current = false
            setIsPlaying(false)
            setError('暂时无法播放，请检查网络后重试。')
        }
    }

    return (
        <div className="music-widget" ref={containerRef}>
            <section
                className="music-panel"
                id="music-panel"
                aria-label="音乐播放器"
                hidden={!isOpen}
            >
                <div className="music-panel-heading">
                    <span className="eyebrow">A LITTLE BACKGROUND MUSIC</span>
                    <button
                        type="button"
                        className="icon-button"
                        aria-label="收起播放器"
                        onClick={() => {
                            setIsOpen(false)
                            triggerRef.current?.focus()
                        }}
                    >
                        <Icon name="close" size={17} />
                    </button>
                </div>
                <div
                    className={`record-art${isPlaying ? ' is-playing' : ''}`}
                    aria-hidden="true"
                >
                    <div className="record-disc">
                        <span>cb.</span>
                    </div>
                    <span className="record-caption">
                        SOUNDTRACK
                        <br />
                        FOR A QUIET MOMENT
                    </span>
                </div>
                <div className="music-meta">
                    <strong>{currentTrack.title}</strong>
                    <span>{currentTrack.artist}</span>
                </div>
                <div className="music-seek">
                    <input
                        type="range"
                        min="0"
                        max={duration || 1}
                        step="1"
                        value={Math.min(currentTime, duration || 1)}
                        disabled={!duration}
                        aria-label="播放进度"
                        onChange={(event) => {
                            const time = Number(event.target.value)
                            if (audioRef.current)
                                audioRef.current.currentTime = time
                            setCurrentTime(time)
                        }}
                    />
                    <div>
                        <span>{formatTime(currentTime)}</span>
                        <span>{formatTime(duration)}</span>
                    </div>
                </div>
                <div className="music-controls">
                    <button
                        type="button"
                        className="icon-button"
                        aria-label="上一首"
                        onClick={() => goToTrack(trackIndex - 1)}
                    >
                        <Icon name="previous" size={19} />
                    </button>
                    <button
                        type="button"
                        className="music-play"
                        aria-label={isPlaying ? '暂停音乐' : '播放音乐'}
                        onClick={() => void togglePlayback()}
                    >
                        <Icon name={isPlaying ? 'pause' : 'play'} size={21} />
                    </button>
                    <button
                        type="button"
                        className="icon-button"
                        aria-label="下一首"
                        onClick={() => goToTrack(trackIndex + 1)}
                    >
                        <Icon name="next" size={19} />
                    </button>
                </div>
                {error && (
                    <p className="music-error" role="status">
                        {error}
                    </p>
                )}
                <div className="music-options">
                    <label>
                        曲目
                        <select
                            value={trackIndex}
                            onChange={(event) =>
                                goToTrack(Number(event.target.value))
                            }
                        >
                            {musicTracks.map((track, index) => (
                                <option value={index} key={track.src}>
                                    {track.title}
                                </option>
                            ))}
                        </select>
                    </label>
                    <label className="music-volume">
                        音量
                        <input
                            type="range"
                            min="0"
                            max="1"
                            step="0.01"
                            value={volume}
                            onChange={(event) =>
                                setVolume(Number(event.target.value))
                            }
                            aria-label="音乐音量"
                        />
                    </label>
                </div>
            </section>
            <button
                ref={triggerRef}
                type="button"
                className={`music-trigger${isPlaying ? ' is-playing' : ''}`}
                aria-expanded={isOpen}
                aria-controls="music-panel"
                onClick={() => setIsOpen(!isOpen)}
            >
                <span className="music-trigger-icon">
                    <Icon name="music" size={17} />
                </span>
                <span>{isPlaying ? currentTrack.title : '来点音乐'}</span>
                {isPlaying ? (
                    <span className="equalizer" aria-hidden="true">
                        <i />
                        <i />
                        <i />
                    </span>
                ) : (
                    <span className="music-trigger-hint">放松一下</span>
                )}
            </button>
            <audio
                ref={audioRef}
                src={currentTrack.src}
                preload="none"
                onEnded={() => {
                    wantsPlayback.current = true
                    goToTrack(trackIndex + 1)
                }}
                onPause={() => setIsPlaying(false)}
                onPlay={() => {
                    setIsPlaying(true)
                    setError('')
                }}
                onTimeUpdate={(event) =>
                    setCurrentTime(event.currentTarget.currentTime)
                }
                onDurationChange={(event) =>
                    setDuration(
                        Number.isFinite(event.currentTarget.duration)
                            ? event.currentTarget.duration
                            : 0,
                    )
                }
                onError={() => {
                    setIsPlaying(false)
                    wantsPlayback.current = false
                    setError('音乐暂时无法加载，请稍后重试。')
                }}
            />
        </div>
    )
}
