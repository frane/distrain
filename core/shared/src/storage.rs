//! S3-compatible storage client for R2.

use anyhow::{Context, Result};
use aws_sdk_s3::primitives::ByteStream;
use aws_sdk_s3::types::{CompletedMultipartUpload, CompletedPart};
use aws_sdk_s3::Client;
use tracing::{info, warn, instrument};

use crate::config::StorageConfig;

/// Async S3 client wrapper for R2 operations.
#[derive(Clone)]
pub struct Storage {
    client: Client,
    bucket: String,
}

impl Storage {
    /// Create a new storage client from config.
    pub async fn new(config: &StorageConfig) -> Result<Self> {
        let sdk_config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .endpoint_url(&config.endpoint)
            .region(aws_config::Region::new(config.region.clone()))
            .credentials_provider(aws_sdk_s3::config::Credentials::new(
                &config.access_key_id,
                &config.secret_access_key,
                None,
                None,
                "distrain",
            ))
            .load()
            .await;

        let s3_config = aws_sdk_s3::config::Builder::from(&sdk_config)
            .force_path_style(true)
            .build();

        let client = Client::from_conf(s3_config);

        Ok(Self {
            client,
            bucket: config.bucket.clone(),
        })
    }

    /// Multipart upload threshold: 100 MB.
    const MULTIPART_THRESHOLD: usize = 100 * 1024 * 1024;
    /// Multipart part size: 100 MB.
    const PART_SIZE: usize = 100 * 1024 * 1024;

    /// Max retries for upload operations.
    const UPLOAD_MAX_RETRIES: u32 = 5;

    /// Upload bytes to a key. Uses multipart upload for data > 100 MB.
    /// Retries up to 5 times with exponential backoff on transient errors.
    #[instrument(skip(self, data), fields(bucket = %self.bucket, key = %key, size = data.len()))]
    pub async fn put(&self, key: &str, data: Vec<u8>) -> Result<()> {
        let len = data.len();
        if len <= Self::MULTIPART_THRESHOLD {
            let mut last_err = None;
            for attempt in 1..=Self::UPLOAD_MAX_RETRIES {
                match self.client
                    .put_object()
                    .bucket(&self.bucket)
                    .key(key)
                    .body(data.clone().into())
                    .send()
                    .await
                {
                    Ok(_) => {
                        info!("Uploaded {} bytes", len);
                        return Ok(());
                    }
                    Err(e) => {
                        warn!("S3 put_object failed (attempt {attempt}/{}): {e}", Self::UPLOAD_MAX_RETRIES);
                        last_err = Some(e);
                        if attempt < Self::UPLOAD_MAX_RETRIES {
                            let backoff = std::time::Duration::from_secs(2u64.pow(attempt));
                            tokio::time::sleep(backoff).await;
                        }
                    }
                }
            }
            return Err(last_err.unwrap()).context("S3 put_object failed after retries");
        } else {
            self.put_multipart(key, &data).await?;
        }
        info!("Uploaded {} bytes", len);
        Ok(())
    }

    /// Multipart upload for large objects.
    async fn put_multipart(&self, key: &str, data: &[u8]) -> Result<()> {
        let create = self
            .client
            .create_multipart_upload()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .context("S3 create_multipart_upload failed")?;

        let upload_id = create
            .upload_id()
            .context("No upload_id in multipart response")?
            .to_string();

        let mut completed_parts = Vec::new();
        let total_parts = (data.len() + Self::PART_SIZE - 1) / Self::PART_SIZE;

        for (i, chunk) in data.chunks(Self::PART_SIZE).enumerate() {
            let part_number = (i + 1) as i32;
            info!(
                "Uploading part {}/{} ({} bytes)",
                part_number,
                total_parts,
                chunk.len()
            );

            let mut part_resp = None;
            for attempt in 1..=Self::UPLOAD_MAX_RETRIES {
                match self
                    .client
                    .upload_part()
                    .bucket(&self.bucket)
                    .key(key)
                    .upload_id(&upload_id)
                    .part_number(part_number)
                    .body(ByteStream::from(chunk.to_vec()))
                    .send()
                    .await
                {
                    Ok(resp) => {
                        part_resp = Some(resp);
                        break;
                    }
                    Err(e) => {
                        warn!("S3 upload_part {part_number} failed (attempt {attempt}/{}): {e}", Self::UPLOAD_MAX_RETRIES);
                        if attempt == Self::UPLOAD_MAX_RETRIES {
                            return Err(e).context(format!("S3 upload_part {} failed after retries", part_number));
                        }
                        let backoff = std::time::Duration::from_secs(2u64.pow(attempt));
                        tokio::time::sleep(backoff).await;
                    }
                }
            }
            let resp = part_resp.unwrap();

            completed_parts.push(
                CompletedPart::builder()
                    .part_number(part_number)
                    .e_tag(resp.e_tag().unwrap_or_default())
                    .build(),
            );
        }

        self.client
            .complete_multipart_upload()
            .bucket(&self.bucket)
            .key(key)
            .upload_id(&upload_id)
            .multipart_upload(
                CompletedMultipartUpload::builder()
                    .set_parts(Some(completed_parts))
                    .build(),
            )
            .send()
            .await
            .context("S3 complete_multipart_upload failed")?;

        Ok(())
    }

    /// Download bytes from a key.
    #[instrument(skip(self), fields(bucket = %self.bucket, key = %key))]
    pub async fn get(&self, key: &str) -> Result<Vec<u8>> {
        let resp = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .context("S3 get_object failed")?;

        let data = resp
            .body
            .collect()
            .await
            .context("Failed to read S3 body")?
            .into_bytes()
            .to_vec();

        info!("Downloaded {} bytes", data.len());
        Ok(data)
    }

    /// Download to a local file.
    #[instrument(skip(self), fields(bucket = %self.bucket, key = %key))]
    pub async fn download_to_file(&self, key: &str, path: &std::path::Path) -> Result<()> {
        let data = self.get(key).await?;
        if let Some(parent) = path.parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        tokio::fs::write(path, &data).await?;
        Ok(())
    }

    /// Upload from a local file.
    #[instrument(skip(self), fields(bucket = %self.bucket, key = %key))]
    pub async fn upload_from_file(&self, key: &str, path: &std::path::Path) -> Result<()> {
        let data = tokio::fs::read(path).await?;
        self.put(key, data).await
    }

    /// Get JSON object, deserialized.
    pub async fn get_json<T: serde::de::DeserializeOwned>(&self, key: &str) -> Result<T> {
        let data = self.get(key).await?;
        serde_json::from_slice(&data).context("Failed to deserialize JSON from S3")
    }

    /// Put JSON object, serialized.
    pub async fn put_json<T: serde::Serialize>(&self, key: &str, value: &T) -> Result<()> {
        let data = serde_json::to_vec_pretty(value)?;
        self.put(key, data).await
    }

    /// Check if a key exists.
    pub async fn exists(&self, key: &str) -> Result<bool> {
        match self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
        {
            Ok(_) => Ok(true),
            Err(e) => {
                let service_err = e.into_service_error();
                if service_err.is_not_found() {
                    Ok(false)
                } else {
                    Err(anyhow::anyhow!("S3 head_object failed: {}", service_err))
                }
            }
        }
    }

    /// Delete an object from S3.
    pub async fn delete(&self, key: &str) -> Result<()> {
        self.client
            .delete_object()
            .bucket(&self.bucket)
            .key(key)
            .send()
            .await
            .context("S3 delete_object failed")?;
        Ok(())
    }

    /// List all keys under a prefix.
    pub async fn list_keys(&self, prefix: &str) -> Result<Vec<String>> {
        let mut keys = Vec::new();
        let mut continuation_token: Option<String> = None;

        loop {
            let mut req = self.client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(prefix);

            if let Some(token) = &continuation_token {
                req = req.continuation_token(token);
            }

            let resp = req.send().await.context("S3 list_objects_v2 failed")?;

            for obj in resp.contents() {
                if let Some(key) = obj.key() {
                    keys.push(key.to_string());
                }
            }

            if resp.is_truncated() == Some(true) {
                continuation_token = resp.next_continuation_token().map(|s| s.to_string());
            } else {
                break;
            }
        }

        Ok(keys)
    }

    /// Create the bucket if it doesn't exist.
    pub async fn ensure_bucket(&self) -> Result<()> {
        // Check if bucket already exists
        match self.client.head_bucket().bucket(&self.bucket).send().await {
            Ok(_) => {
                info!("Bucket already exists: {}", self.bucket);
                return Ok(());
            }
            Err(_) => {
                // Bucket doesn't exist (or access error), try to create
            }
        }

        match self
            .client
            .create_bucket()
            .bucket(&self.bucket)
            .send()
            .await
        {
            Ok(_) => {
                info!("Created bucket: {}", self.bucket);
                Ok(())
            }
            Err(e) => {
                // Tolerate "already exists" in any form
                let msg = e.to_string();
                if msg.contains("BucketAlready") || msg.contains("already") {
                    info!("Bucket already exists: {}", self.bucket);
                    Ok(())
                } else {
                    Err(anyhow::anyhow!("Failed to create bucket: {}", msg))
                }
            }
        }
    }
}
