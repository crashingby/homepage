export type MusicTrack = {
    title: string
    artist?: string
    src: string
}

const musicBaseUrl = `https://github.com/crashingby/homepage/releases/download/music-assets-v1/`

export const musicTracks: MusicTrack[] = [
    // 把音频文件放到 public/music/ 目录后，在这里添加歌曲。
    {
        title: 'Tractatus Logico-philosophicus',
        artist: '松本文紀',
        src: `${musicBaseUrl}Tractatus-Logico-philosophicus.ogg`,
    },
    {
        title: '夏の大三角',
        artist: 'ryo',
        src: `${musicBaseUrl}Summer-triangle.ogg`,
    },
]

export { musicBaseUrl }
