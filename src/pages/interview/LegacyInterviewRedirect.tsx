import { Navigate, useLocation, useParams } from 'react-router-dom'
import { getLegacyInterviewPath } from '../../lib/interview'

export function LegacyInterviewRedirect() {
    const { slug } = useParams()
    const { search, hash } = useLocation()
    const pathname = (slug && getLegacyInterviewPath(slug)) || '/interview'
    return <Navigate to={{ pathname, search, hash }} replace />
}
