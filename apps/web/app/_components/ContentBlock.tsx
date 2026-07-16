import "./content.css"
import GlassCard from "./GlassCard"

export default function ContentBlock(props: any)
{
    return (
        <div className="content-block ">
            <GlassCard>
                {props.children}
            </GlassCard>
        </div>
    )
}