import { useEffect, useRef, useState } from 'react'
import { musicTracks } from '../data/music'

export function MusicPlayer() {
    const audioRef = useRef<HTMLAudioElement>(null)
    const trackMenuRef = useRef<HTMLDivElement>(null)
    const [trackIndex, setTrackIndex] = useState(0)
    const [isPlaying, setIsPlaying] = useState(false)
    const [isTrackMenuOpen, setIsTrackMenuOpen] = useState(false)
    const [volume, setVolume] = useState(0.72)

    const currentTrack = musicTracks[trackIndex]

    useEffect(() => {
        const audio = audioRef.current

        if (!audio || !currentTrack) {
            return
        }

        audio.load()

        if (isPlaying) {
            void audio.play().catch(() => setIsPlaying(false))
        }
    }, [currentTrack, isPlaying])

    useEffect(() => {
        const audio = audioRef.current

        if (audio) {
            audio.volume = volume
        }
    }, [volume])

    useEffect(() => {
        const closeTrackMenu = (event: MouseEvent) => {
            if (!trackMenuRef.current?.contains(event.target as Node)) {
                setIsTrackMenuOpen(false)
            }
        }

        document.addEventListener('mousedown', closeTrackMenu)

        return () => {
            document.removeEventListener('mousedown', closeTrackMenu)
        }
    }, [])

    if (musicTracks.length === 0 || !currentTrack) {
        return null
    }

    const goToTrack = (nextIndex: number) => {
        const normalizedIndex = (nextIndex + musicTracks.length) % musicTracks.length
        setTrackIndex(normalizedIndex)
    }

    const togglePlayback = async () => {
        const audio = audioRef.current

        if (!audio) {
            return
        }

        if (isPlaying) {
            audio.pause()
            setIsPlaying(false)
            return
        }

        try {
            await audio.play()
            setIsPlaying(true)
        } catch {
            setIsPlaying(false)
        }
    }

    return (
        <section className="music-player" aria-label="Homepage music player">
            <div className="music-player__meta">
                <span className="music-player__label">Music</span>
                <strong>{currentTrack.title}</strong>
                {currentTrack.artist ? <span>{currentTrack.artist}</span> : null}
            </div>

            <div className="music-player__controls">
                <button type="button" onClick={() => goToTrack(trackIndex - 1)}>
                    上一首
                </button>
                <button type="button" className="music-player__primary" onClick={togglePlayback}>
                    {isPlaying ? '暂停' : '播放'}
                </button>
                <button type="button" onClick={() => goToTrack(trackIndex + 1)}>
                    下一首
                </button>
            </div>

            <div className="music-player__track-picker" ref={trackMenuRef}>
                <span>曲目</span>
                <button
                    type="button"
                    className="music-player__track-trigger"
                    aria-haspopup="listbox"
                    aria-expanded={isTrackMenuOpen}
                    onClick={() => setIsTrackMenuOpen((isOpen) => !isOpen)}
                >
                    <span>{currentTrack.title}</span>
                </button>

                {isTrackMenuOpen ? (
                    <div className="music-player__track-menu" role="listbox" aria-label="选择歌曲">
                        {musicTracks.map((track, index) => (
                            <button
                                key={track.src}
                                type="button"
                                className={index === trackIndex ? 'is-active' : undefined}
                                role="option"
                                aria-selected={index === trackIndex}
                                onClick={() => {
                                    setTrackIndex(index)
                                    setIsTrackMenuOpen(false)
                                }}
                            >
                                <strong>{track.title}</strong>
                                {track.artist ? <span>{track.artist}</span> : null}
                            </button>
                        ))}
                    </div>
                ) : null}
            </div>

            <label className="music-player__volume">
                <span>音量 {Math.round(volume * 100)}%</span>
                <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.01"
                    value={volume}
                    onChange={(event) => setVolume(Number(event.target.value))}
                    aria-label="调节音乐音量"
                />
            </label>

            <audio
                ref={audioRef}
                src={currentTrack.src}
                onEnded={() => goToTrack(trackIndex + 1)}
                onPause={() => setIsPlaying(false)}
                onPlay={() => setIsPlaying(true)}
            />
        </section>
    )
}
