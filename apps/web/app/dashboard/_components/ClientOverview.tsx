import { Card } from "@/components/ui/card";
import { PageHeading } from "./presentation";

/** Separate extension point for the future client workspace. */
export default function ClientOverview() {
  return (
    <>
      <PageHeading
        title="Your workspace"
        description="Your personal space for observations and forecasts."
      />
      <Card className="p-6">
        <h2 className="font-medium">Client dashboard is coming soon</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Your account is ready. Available sections appear in the navigation.
        </p>
      </Card>
    </>
  );
}
