"use client";
import { useEffect, useState } from "react";
import { Loader2, Pencil, Plus, ShieldCheck, UsersRound } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Card } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import {
  EmptyState,
  Message,
  PageHeading,
  Pagination,
  selectClass,
  TableLoading,
} from "../_components/presentation";
import { dashboardRequest, useSession, type User } from "../session";

type Group = { name: string; permissions: string[] };
export default function Users() {
  const actor = useSession();
  const [users, setUsers] = useState<User[]>([]);
  const [groups, setGroups] = useState<Group[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [version, setVersion] = useState(0);
  const [editing, setEditing] = useState<User | null>(null);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState("");
  const [formError, setFormError] = useState("");
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let disposed = false;
    Promise.all([
      dashboardRequest<{ items: User[]; total: number }>(`/users?page=${page}`),
      dashboardRequest<Group[]>("/groups"),
    ])
      .then(([result, available]) => {
        if (!disposed) {
          setUsers(result.items);
          setTotal(result.total);
          setGroups(available);
          setError("");
        }
      })
      .catch((e) => {
        if (!disposed) {
          setUsers([]);
          setError(e.message);
        }
      })
      .finally(() => {
        if (!disposed) setLoading(false);
      });
    return () => {
      disposed = true;
    };
  }, [page, version]);
  function edit(user: User | null) {
    setEditing(user);
    setFormError("");
    setNotice("");
    setOpen(true);
  }
  return (
    <>
      <PageHeading
        title="Users & access"
        description="Manage workspace accounts and the groups that control their access."
        action={
          <Button disabled={loading || !!error} onClick={() => edit(null)}>
            <Plus className="mr-2 size-4" />
            Create user
          </Button>
        }
      />
      {error && (
        <Message>
          {error}{" "}
          <Button
            variant="link"
            className="h-auto p-0 text-inherit"
            onClick={() => {
              setLoading(true);
              setVersion((v) => v + 1);
            }}
          >
            Retry
          </Button>
        </Message>
      )}
      {notice && <Message success>{notice}</Message>}
      <Card className="overflow-hidden shadow-none">
        <div className="flex items-center justify-between border-b px-5 py-4">
          <div className="flex items-center gap-3">
            <UsersRound className="size-4 text-muted-foreground" />
            <h2 className="text-sm font-semibold">Workspace members</h2>
            <Badge variant="secondary" className="font-normal">
              {total}
            </Badge>
          </div>
          <span className="hidden text-xs text-muted-foreground sm:block">
            Access is assigned through groups
          </span>
        </div>
        {loading ? (
          <TableLoading />
        ) : users.length ? (
          <Table>
            <TableHeader>
              <TableRow className="bg-muted/40 hover:bg-muted/40">
                <TableHead className="px-5">User</TableHead>
                <TableHead>Groups</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="px-5 text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {users.map((user) => (
                <TableRow key={user.id}>
                  <TableCell className="px-5 py-4">
                    <div className="flex items-center gap-3">
                      <div className="flex size-9 items-center justify-center rounded-full border bg-muted text-xs font-medium uppercase text-muted-foreground">
                        {user.username.slice(0, 2)}
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium">{user.username}</span>
                        {user.id === actor?.id && (
                          <span className="text-xs text-muted-foreground">
                            (you)
                          </span>
                        )}
                      </div>
                    </div>
                  </TableCell>
                  <TableCell>
                    <div className="flex flex-wrap gap-1.5">
                      {user.groups.length ? (
                        user.groups.map((group) => (
                          <Badge
                            key={group}
                            variant="outline"
                            className="gap-1.5 font-normal"
                          >
                            <ShieldCheck className="size-3 text-primary" />
                            {group}
                          </Badge>
                        ))
                      ) : (
                        <span className="text-xs text-muted-foreground">
                          No groups
                        </span>
                      )}
                    </div>
                  </TableCell>
                  <TableCell>
                    <Badge
                      variant="outline"
                      className={
                        user.active
                          ? "border-emerald-500/20 bg-emerald-500/5 font-normal text-emerald-300"
                          : "border-muted-foreground/20 font-normal text-muted-foreground"
                      }
                    >
                      <span
                        className={`mr-1.5 size-1.5 rounded-full ${user.active ? "bg-emerald-400" : "bg-muted-foreground"}`}
                      />
                      {user.active ? "Active" : "Blocked"}
                    </Badge>
                  </TableCell>
                  <TableCell className="px-5 text-right">
                    <Button
                      variant="ghost"
                      size="sm"
                      aria-label={`Edit ${user.username}`}
                      onClick={() => edit(user)}
                    >
                      <Pencil className="mr-2 size-3.5" />
                      Edit
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        ) : (
          <EmptyState
            title="No users to display"
            description={
              error
                ? "Retry loading the member list."
                : "Create an account to give someone access to this workspace."
            }
          />
        )}
        <Pagination
          page={page}
          size={50}
          total={total}
          busy={loading}
          onChange={(value) => {
            setLoading(true);
            setPage(value);
          }}
        />
      </Card>
      <div className="flex gap-3 rounded-lg border border-dashed p-4">
        <ShieldCheck className="mt-0.5 size-4 text-muted-foreground" />
        <p className="text-xs leading-relaxed text-muted-foreground">
          Only administrators can create accounts. Users without assigned
          permissions can sign in but cannot access dashboard data.
        </p>
      </div>
      <Dialog
        open={open}
        onOpenChange={(value) => {
          if (!busy) setOpen(value);
        }}
      >
        <DialogContent className="max-h-[90svh] w-[calc(100%-2rem)] overflow-y-auto rounded-xl sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{editing ? "Edit user" : "Create user"}</DialogTitle>
            <DialogDescription>
              {editing
                ? `Update the account and access for ${editing.username}.`
                : "Add a member to your Argus workspace."}
            </DialogDescription>
          </DialogHeader>
          <form
            key={editing?.id ?? "new"}
            className="space-y-5 pt-2"
            onSubmit={async (e) => {
              e.preventDefault();
              const values = new FormData(e.currentTarget);
              setBusy(true);
              setFormError("");
              setNotice("");
              const password = String(values.get("password") || "");
              try {
                const selected = values.getAll("groups").map(String);
                await dashboardRequest(
                  editing ? `/users/${editing.id}` : "/users",
                  editing ? "PATCH" : "POST",
                  editing
                    ? {
                        active: values.get("active") === "true",
                        ...(editing.id === actor?.id
                          ? {}
                          : { groups: selected }),
                        ...(password ? { password } : {}),
                      }
                    : {
                        username: values.get("username"),
                        password,
                        groups: selected,
                      },
                );
                if (editing?.id === actor?.id && password) {
                  window.location.assign("/dashboard/login");
                  return;
                }
                setNotice(
                  editing
                    ? "User updated. Changes to access or passwords revoke existing sessions."
                    : "User created successfully.",
                );
                setOpen(false);
                setEditing(null);
                setLoading(true);
                setVersion((v) => v + 1);
              } catch (e) {
                setFormError(
                  e instanceof Error ? e.message : "Could not save user",
                );
              } finally {
                setBusy(false);
              }
            }}
          >
            {!editing && (
              <div className="space-y-2">
                <Label htmlFor="new-username">Username</Label>
                <Input
                  id="new-username"
                  name="username"
                  required
                  maxLength={80}
                  pattern="[a-zA-Z0-9_.@\-]+"
                  autoComplete="off"
                  placeholder="e.g. alex"
                />
              </div>
            )}
            <div className="space-y-2">
              <Label htmlFor="new-password">
                {editing ? "New password" : "Password"}
                {editing && (
                  <span className="ml-1 font-normal text-muted-foreground">
                    (optional)
                  </span>
                )}
              </Label>
              <Input
                id="new-password"
                name="password"
                type="password"
                minLength={12}
                maxLength={256}
                required={!editing}
                autoComplete="new-password"
              />
              <p className="text-xs text-muted-foreground">
                Use 12–256 characters.
                {editing ? " Leave empty to keep the current password." : ""}
              </p>
            </div>
            <fieldset className="space-y-3">
              <legend className="mb-2 text-sm font-medium">Groups</legend>
              {groups.map((group) => (
                <label
                  key={group.name}
                  className="flex cursor-pointer items-center gap-3 rounded-lg border p-3"
                >
                  <Checkbox
                    name="groups"
                    value={group.name}
                    defaultChecked={editing?.groups.includes(group.name)}
                    disabled={editing?.id === actor?.id}
                  />
                  <span className="text-sm">{group.name}</span>
                </label>
              ))}
              {editing?.id === actor?.id && (
                <p className="text-xs text-muted-foreground">
                  You cannot remove your own administrator access.
                </p>
              )}
            </fieldset>
            {editing && (
              <div className="space-y-2">
                <Label htmlFor="user-status">Account status</Label>
                <select
                  id="user-status"
                  name="active"
                  defaultValue={String(editing.active)}
                  className={selectClass}
                >
                  <option value="true">Active</option>
                  <option value="false" disabled={editing.id === actor?.id}>
                    Blocked
                  </option>
                </select>
              </div>
            )}
            {formError && <Message>{formError}</Message>}
            <DialogFooter className="gap-2 border-t pt-4">
              <Button
                type="button"
                variant="outline"
                disabled={busy}
                onClick={() => setOpen(false)}
              >
                Cancel
              </Button>
              <Button disabled={busy}>
                {busy && <Loader2 className="mr-2 size-4 animate-spin" />}
                {busy ? "Saving…" : editing ? "Save changes" : "Create user"}
              </Button>
            </DialogFooter>
          </form>
        </DialogContent>
      </Dialog>
    </>
  );
}
