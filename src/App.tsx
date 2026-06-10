import './App.css'
import { Hero } from './components/Hero'
import { About } from './components/About'
import { ProjectPreview } from './components/ProjectPreview'

function App() {
    return (
        <main className="app">
            <Hero />
            <About />
            <ProjectPreview />
        </main>
    )
}

export default App