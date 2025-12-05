# Agentic RAG: Role-aware access

Role-aware access is the ability for the RAG to give answers based on the asking user's permissions on the documents that are stored. Not all users may have access to all the documents, and setting up a role-aware system is key to allow the RAG to answer accordingly to each user.

Consists of two main parts:

- **Ingestion step**: when ingesting documents, metadata relevant to access is added to the document so that access policy is tied to each document itself. Which metadata is added to the document depends on the strategy chosen to handle access, for example, role names or tags
- **Retrieval**: depending on the way we want the system to behave when a user tries to access documents without permission, the retrieval step could either discard all non-available documents and act as if they don't exist, or handle permissions in a way so the user knows they don't have access to certain documents.

## Role-Aware Access: Ingestion and Authorization

The most robust way to handle access control in RAG is to enforce it at the **Retrieval stage**, before the document context even reaches the LLM.

### 1. Document Ingestion (Metadata Tagging)

During the ingestion step, metadata is attached to every document chunk that specifies which roles can access it.

* **Process:** After converting the document (text documents, audio transcripts) into chunks and generating embeddings, add the access control information to the chunk's metadata before storing it in the VectorDB.
**Example Metadata:**
    * `document_id`: `policy_HR_005`
    * `required_role`: `['HR_Manager', 'Admin']`  _-- here users have roles that give them access to documents_
    * `tier_level`: `Tier 1`  _-- any useful metadata can be added to the document at the ingestion_
    * `client_group`: `['Internal']`
    * `sensitivity`: `Confidential`

### 2. User Authorization (Session Management)

When a user logs into your system and asks a query, the system must:

1.  **Authenticate** the user (verify their identity).
2.  **Authorize** the user: Map their identity to a set of active **Roles** and **Permissions** (e.g., User Jane has the roles: `HR_Manager`, `Employee`, and the permission `read_confidential`).
3.  **Create a Contextual Filter:** This set of permissions is converted into a **VectorDB filter query** (e.g., `role IN ('HR_Manager', 'Employee')`).

## Role Access Handling Strategies

There are two main strategies for dealing with documents a user is not authorized to access. The choice affects the user experience and the complexity of the Agent LLM's prompt.

### Strategy 1: Act Like the Document Doesn't Exist (Filter-Based Retrieval)

This is the standard and **most secure** approach.

* **Mechanism:** The system ensures that the **VectorDB query includes the security filter** derived from the user's role *before* the search is executed.
    * *Search Query:* Find chunks similar to "Q" **AND** where `required_role` is in `user_roles`.
* **Outcome for the Agent:** The LLM only ever receives context the user is allowed to see. The missing information is invisible to the Agent and, therefore, cannot be mentioned in the final response.
* **User Experience (UX):**
    * **If the answer exists in accessible documents:** The agent answers normally
    * **If the answer only exists in restricted documents:** The agent answers: "I cannot find information regarding X," or "The available documentation does not specify X."
* **Pros:** **Highest security** (enforcement happens at the database level). Simplified Agent LLM task (no need to reason about permissions)
* **Cons:** Users may not know *why* the information is missing.

### Strategy 2: Acknowledge Existence but Deny Access (Complex)

This approach is sometimes desired for **transparency**, but it adds significant complexity and risk.

* **Mechanism:** The retrieval process works in two steps:
    1.  **Retrieval:** The Agent retrieves *all* potentially relevant chunks (including restricted ones)
    2.  **Filtering & Tagging:** The Orchestrator code marks each retrieved chunk with a flag (i.e: `[Access: Granted]` or `[Access: Denied]`)
    3.  **Agent Reasoning:** The Agent LLM (Planner) receives all chunks and must be explicitly instructed to read the access flag
* **Outcome for the Agent:** The Agent receives both the content of the permitted documents and the metadata/ID of the restricted documents
* **User Experience (UX):**
    * The agent answers: "I found a document, **policy_HR_005**, that addresses your question, but your current **HR_Employee** role does not grant you access to its contents."
* **Pros:** **High Transparency**. Users know exactly what is missing and why
* **Cons:** **High Security Risk** (requires moving restricted *content* closer to the LLM). Greatly increases the **Agent LLM's complexity** and unreliability, as the model must be perfectly prompted to suppress sensitive content while mentioning its ID. This is a common failure point for **prompt injection** or hallucination

## Agentic RAG and Role-Aware Access

The Agent LLM's role-aware capabilities depend entirely on which of the two strategies above is used:

### Strategy 1 (Filter-Based Retrieval)

The Agent's loop is simplified and secure:

1.  **Planner:** Receives question + **User's Filter** (`role=Admin` or `tags=[people, internal, documentation]`)
2.  **Action:** Calls `search_vector_db` with the question and the attached filter
3.  **Observation:** Receives only permitted context
4.  **Reflection:** If context is missing or is not accessible due to access restrictions, it concludes there is no internal answer
5.  **Generation:** Answers based only on the available context

### Strategy 2 (Transparency/Metadata-Tagging)

The Agent must be robust enough to handle the security logic:

1.  **Planner:** Receives question.
2.  **Action:** Calls `search_vector_db` (potentially without filter, risking data exposure).
3.  **Observation:** Receives permitted *and* denied context, all with access tags.
4.  **Reflection:** The Agent must internally reason: "I see three documents. Doc A is permitted, Doc B is denied. I will synthesize A and inform the user about the existence of B without revealing its content."
5.  **Generation:** Synthesizes the final answer *and* generates the access denial message.

## Which strategy to use?

It all depends mainly on whether we want to give the RAG the ability to let users know if documents they don't have access to exist.

For compliance and security, **Strategy 1 (Filter-Based Retrieval)** is strongly recommended. It is simpler to implement using features common to VectorDBs, and relies on code (the secure filter) rather than the probabilistic reasoning of the LLM for security enforcement.

---

## Role-Aware Access Strategies: Rigid Roles vs. Tagging System

Implementing a **Role-Aware Access** system within a **RAG (Retrieval-Augmented Generation)** architecture requires security information to be consistently enforced during document ingestion and retrieval. We will explore two primary approaches for mapping user permissions to documents: **Rigid Roles** and the **Tagging System (ABAC)**, along with a hybrid strategy combining both.


### 1. Rigid Roles (Classic Role-Based Access Control - RBAC)

This is the most straightforward approach, using the user's primary and secondary **roles** to directly dictate their access.

#### Mechanism

1.  **Ingestion:** Each document chunk is tagged directly with the **exact role(s)** required for access (e.g., `required_role: 'HR_Manager'` or `required_roles: ['Finance', 'Admin']`).
2.  **Authorization:** The system checks if the user's assigned role(s) **exactly match** or are included in the document's required roles.

#### Pros

* **Simplicity:** The logic is linear and easy to implement in the initial authorization layer.
* **Easy Auditing:** Simple to track who has access to which documents based on a single, well-defined column.

#### Cons

* **Rigidity:** If a document is relevant to multiple departments (e.g., a security manual is for both `Finance` and `IT`), you must tag it with an encompassing role or list all roles, which can lead to complex maintenance.
* **Maintenance Overhead:** Creating a new role often requires manually updating the access metadata for many existing documents.

---

### 2. Tagging System (Attribute-Based Access Control - ABAC)

This approach is highly flexible, decoupling the document from a specific role and basing access on **descriptive attributes** or **classifications (tags)**.

#### Mechanism

1.  **Ingestion (Tagging):** Document chunks are tagged with a list of **attributes** (e.g., `access_tags: ['Finance', 'Confidential', 'Audit_2024']`).
2.  **Permission Mapping:** An external **Permission Mapping Table** is maintained, defining which **roles** are authorized to read which **tags** (e.g., the `Finance Auditor` role is allowed to read the tags `Finance` and `Confidential`).
3.  **Authorization (Intersection):** The system aggregates all tags a user is permitted to read, forming the **User Access Tag Set**. Retrieval then uses set intersection logic: *Retrieve chunks where the document's `access_tags` array has at least one item that intersects with the user's `User Access Tag Set`.*

#### Advantages

* **Flexibility & Granularity:** A single document can be accessed by diverse roles simultaneously based on the attributes they are authorized for.
* **Scalability:** When a new role is created, you only update the central permission mapping table; you do not need to retag the entire document corpus.
* **Decoupling:** Security is managed by abstract attributes, not specific job titles.

#### Disadvantages

* **Initial Complexity:** Requires a separate authorization layer to map user roles to the permitted tag sets.

---

### 3. Hybrid Approach: Combining Roles and Tags

The most secure and flexible strategy involves combining both rigid role requirements and flexible tagging.

#### Mechanism

A single document chunk contains **multiple access fields** in its metadata:

1.  `required_role_strict`: A single, critical role that *must* have access (e.g., `Admin`).
2.  `access_tags`: An array of flexible tags (e.g., `['Sales', 'Training', 'EU']`).

#### Interaction and Filtering

When the user queries the Agent, the system constructs a complex Boolean filter for the VectorDB:

$$\text{Retrieve where } ((\text{document}.\text{required\_role\_strict} == \text{user}.\text{primary\_role}) \mathbf{OR} (\text{document}.\text{access\_tags} \cap \text{user}.\text{authorized\_tags} \ne \emptyset))$$

#### Benefits

* **Maximum Control:** Allows for documents with critical security needs to be locked down to a single rigid role, while general compliance documents use the flexible tagging system.
* **Resilience:** Provides a layered defense; if one authorization system (e.g., tags) is misconfigured, the strict role check can still protect critical data.
* **Users not locked to roles:** you can create users that don't necessarily adjust to any specific role, but can have access to a set of specific tags

## Security Mandate: Handling Denied Access

In all access control strategies, security must be enforced via **Filter-Based Retrieval**.

* **The system must ensure that the user's permissions are applied as a hard filter on the VectorDB query.**
* When a user does not have permission for a document that holds the answer, the Agent RAG must **act as if the document does not exist.**
* The Agent should simply reply that it **"found no information in the available documentation"** or "cannot provide a specific answer," thus maintaining security and adhering to the principle of least privilege.

# Diagram

The following diagram demostrates the use of roles + tags, where some documents

<img alt="Role + tag based access" src="../resources/role-tag-based-access.png" width="600" height="300">
