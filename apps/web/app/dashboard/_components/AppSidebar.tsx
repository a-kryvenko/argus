"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  ChevronsUpDown,
  ExternalLink,
  LogOut,
  ShieldCheck,
} from "lucide-react";
import {
  Sidebar,
  SidebarContent,
  SidebarFooter,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
  SidebarRail,
  useSidebar,
} from "@/components/ui/sidebar";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { navigation } from "../navigation";
import type { User } from "../session";

export function AppSidebar({
  user,
  logout,
  signingOut,
}: {
  user: User;
  logout: () => void;
  signingOut: boolean;
}) {
  const path = usePathname().replace(/\/$/, "");
  const { isMobile, setOpenMobile } = useSidebar();
  return (
    <Sidebar collapsible="icon" variant="inset">
      <SidebarHeader className="py-4 group-data-[collapsible=icon]:hidden">
        <SidebarMenu>
          <SidebarMenuItem>
            <SidebarMenuButton
              asChild
              size="lg"
              className="hover:bg-transparent"
            >
              <Link href="/dashboard" onClick={() => setOpenMobile(false)}>
                <div className="grid flex-1 text-left leading-tight">
                  <span className="font-semibold tracking-wide text-foreground">
                    ARGUS
                  </span>
                  <span className="text-xs text-muted-foreground">
                    Sunwatch workspace
                  </span>
                </div>
              </Link>
            </SidebarMenuButton>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarHeader>
      <SidebarContent>
        {navigation.map((group) => {
          const items = group.items.filter(
            (item) =>
              !item.permission || user.permissions.includes(item.permission),
          );
          return (
            items.length > 0 && (
              <SidebarGroup key={group.title}>
                <SidebarGroupLabel className="text-[11px] uppercase tracking-widest">
                  {group.title}
                </SidebarGroupLabel>
                <SidebarGroupContent>
                  <SidebarMenu>
                    {items.map((item) => (
                      <SidebarMenuItem key={item.href}>
                        <SidebarMenuButton
                          asChild
                          tooltip={item.title}
                          isActive={path === item.href}
                          className="h-10 data-[active=true]:text-primary"
                        >
                          <Link
                            href={item.href}
                            aria-current={
                              path === item.href ? "page" : undefined
                            }
                            onClick={() => setOpenMobile(false)}
                          >
                            <item.icon className="size-4" />
                            <span>{item.title}</span>
                          </Link>
                        </SidebarMenuButton>
                      </SidebarMenuItem>
                    ))}
                  </SidebarMenu>
                </SidebarGroupContent>
              </SidebarGroup>
            )
          );
        })}
        <SidebarGroup className="mt-auto">
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton asChild tooltip="Public website">
                  <Link href="/">
                    <ExternalLink className="size-4" />
                    <span>Public website</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
      <SidebarFooter className="border-t border-sidebar-border pt-3">
        <SidebarMenu>
          <SidebarMenuItem>
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <SidebarMenuButton
                  size="lg"
                  className="data-[state=open]:bg-sidebar-accent"
                  aria-label="Account menu"
                >
                  <div className="flex size-8 shrink-0 items-center justify-center rounded-lg border bg-muted text-xs font-semibold uppercase">
                    {user.username.slice(0, 2)}
                  </div>
                  <div className="grid flex-1 text-left">
                    <span className="truncate font-medium text-foreground">
                      {user.username}
                    </span>
                    <span className="truncate text-xs text-muted-foreground">
                      {user.groups.join(", ") || "Member"}
                    </span>
                  </div>
                  <ChevronsUpDown className="ml-auto size-4" />
                </SidebarMenuButton>
              </DropdownMenuTrigger>
              <DropdownMenuContent
                className="w-56"
                side={isMobile ? "bottom" : "right"}
                align="end"
                sideOffset={8}
              >
                <DropdownMenuLabel>
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="size-4 text-primary" />
                    {user.username}
                  </div>
                </DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem disabled={signingOut} onSelect={logout}>
                  <LogOut className="mr-2 size-4" />
                  {signingOut ? "Signing out…" : "Sign out"}
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </SidebarMenuItem>
        </SidebarMenu>
      </SidebarFooter>
      <SidebarRail />
    </Sidebar>
  );
}
