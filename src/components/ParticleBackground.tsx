import { useEffect, useRef } from 'react'

type Particle = {
    x: number
    y: number
    vx: number
    vy: number
    radius: number
    alpha: number
}

type BurstParticle = Particle & {
    life: number
    maxLife: number
    hue: number
}

const PARTICLE_COUNT = 72
const LINK_DISTANCE = 130
const MAX_DPR = 2

function randomBetween(min: number, max: number) {
    return min + Math.random() * (max - min)
}

function createParticle(width: number, height: number): Particle {
    return {
        x: Math.random() * width,
        y: Math.random() * height,
        vx: randomBetween(-0.18, 0.18),
        vy: randomBetween(-0.14, 0.14),
        radius: randomBetween(0.8, 1.8),
        alpha: randomBetween(0.28, 0.62),
    }
}

function createBurstParticle(x: number, y: number): BurstParticle {
    const angle = Math.random() * Math.PI * 2
    const speed = randomBetween(1.2, 4.2)
    const maxLife = randomBetween(28, 48)

    return {
        x,
        y,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        radius: randomBetween(1.2, 2.4),
        alpha: 1,
        life: maxLife,
        maxLife,
        hue: randomBetween(176, 204),
    }
}

export function ParticleBackground() {
    const canvasRef = useRef<HTMLCanvasElement>(null)

    useEffect(() => {
        const canvasElement = canvasRef.current
        const canvasContext = canvasElement?.getContext('2d')

        if (!canvasElement || !canvasContext) {
            return
        }

        const canvas = canvasElement
        const context = canvasContext

        const particles: Particle[] = []
        const bursts: BurstParticle[] = []
        let width = window.innerWidth
        let height = window.innerHeight
        let animationFrame = 0

        function resizeCanvas() {
            const dpr = Math.min(window.devicePixelRatio || 1, MAX_DPR)
            width = window.innerWidth
            height = window.innerHeight
            canvas.width = Math.floor(width * dpr)
            canvas.height = Math.floor(height * dpr)
            canvas.style.width = `${width}px`
            canvas.style.height = `${height}px`
            context.setTransform(dpr, 0, 0, dpr, 0, 0)
        }

        function seedParticles() {
            particles.length = 0

            for (let index = 0; index < PARTICLE_COUNT; index += 1) {
                particles.push(createParticle(width, height))
            }
        }

        function drawBackgroundParticles() {
            for (const particle of particles) {
                particle.x += particle.vx
                particle.y += particle.vy

                if (particle.x < -20) particle.x = width + 20
                if (particle.x > width + 20) particle.x = -20
                if (particle.y < -20) particle.y = height + 20
                if (particle.y > height + 20) particle.y = -20

                context.beginPath()
                context.fillStyle = `rgba(125, 211, 252, ${particle.alpha})`
                context.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2)
                context.fill()
            }
        }

        function drawLinks() {
            for (let first = 0; first < particles.length; first += 1) {
                for (let second = first + 1; second < particles.length; second += 1) {
                    const a = particles[first]
                    const b = particles[second]
                    const dx = a.x - b.x
                    const dy = a.y - b.y
                    const distance = Math.hypot(dx, dy)

                    if (distance > LINK_DISTANCE) {
                        continue
                    }

                    const alpha = (1 - distance / LINK_DISTANCE) * 0.12
                    context.beginPath()
                    context.strokeStyle = `rgba(125, 211, 252, ${alpha})`
                    context.lineWidth = 1
                    context.moveTo(a.x, a.y)
                    context.lineTo(b.x, b.y)
                    context.stroke()
                }
            }
        }

        function drawBursts() {
            for (let index = bursts.length - 1; index >= 0; index -= 1) {
                const particle = bursts[index]
                const progress = particle.life / particle.maxLife

                particle.x += particle.vx
                particle.y += particle.vy
                particle.vx *= 0.97
                particle.vy *= 0.97
                particle.life -= 1

                context.beginPath()
                context.fillStyle = `hsla(${particle.hue}, 90%, 68%, ${progress})`
                context.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2)
                context.fill()

                if (particle.life <= 0) {
                    bursts.splice(index, 1)
                }
            }
        }

        function animate() {
            context.clearRect(0, 0, width, height)
            drawLinks()
            drawBackgroundParticles()
            drawBursts()
            animationFrame = window.requestAnimationFrame(animate)
        }

        function handleClick(event: MouseEvent) {
            for (let index = 0; index < 22; index += 1) {
                bursts.push(createBurstParticle(event.clientX, event.clientY))
            }
        }

        function handleResize() {
            resizeCanvas()
            seedParticles()
        }

        resizeCanvas()
        seedParticles()
        animate()

        window.addEventListener('click', handleClick)
        window.addEventListener('resize', handleResize)

        return () => {
            window.cancelAnimationFrame(animationFrame)
            window.removeEventListener('click', handleClick)
            window.removeEventListener('resize', handleResize)
        }
    }, [])

    return <canvas aria-hidden="true" className="particle-canvas" ref={canvasRef} />
}
