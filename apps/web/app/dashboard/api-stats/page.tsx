import { PageHeading } from "../_components/presentation";
import ApiActivity from "../_components/ApiActivity";
export default function ApiStats() {
  return (
    <>
      <PageHeading
        title="API statistics"
        description="Monitor traffic, response times and errors across your API."
      />
      <ApiActivity />
    </>
  );
}
