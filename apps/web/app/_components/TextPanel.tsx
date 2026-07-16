import "./content.css"

export default function TextPanel(props: any)
{
    return (
        <div className="text-panel">
            {props.children}
        </div>
        
    )
}