import "./glass.css"

export default function GlassCard(props: any)
{
    return (
        <div className="glass-card">
            {props.children}
        </div>
    )
}