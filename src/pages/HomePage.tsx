import { About } from '../components/About'
import { Hero } from '../components/Hero'
import { MusicPlayer } from '../components/MusicPlayer'
import { ProjectPreview } from '../components/ProjectPreview'

export function HomePage() {
    return (
        <div className="home-page">
            <Hero />
            <MusicPlayer />
            <About />
            <ProjectPreview />
        </div>
    )
}
