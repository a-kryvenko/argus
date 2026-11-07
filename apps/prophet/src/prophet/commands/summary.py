"""Publish the public summary and its SVGs to S3."""
async def run(args):
    from prophet.services.summary import publish_summary
    await publish_summary()
