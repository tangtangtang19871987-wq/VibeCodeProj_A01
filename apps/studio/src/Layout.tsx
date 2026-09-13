import type { PropsWithChildren } from "react";
import { NavLink } from "react-router-dom";
import { HealthBadge } from "./components/HealthBadge.js";

const navItems = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/memories", label: "Memory Explorer" },
  { to: "/sessions", label: "Sessions" },
  { to: "/review-queue", label: "Review Queue" },
  { to: "/settings", label: "Settings" },
];

export function Layout({ children }: PropsWithChildren) {
  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="app-header__title">Local Agent Memory Studio</div>
        <HealthBadge />
      </header>
      <div className="app-body">
        <nav className="app-nav">
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={item.end}
              className={({ isActive }) =>
                "app-nav__link" + (isActive ? " app-nav__link--active" : "")
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <main className="app-main">{children}</main>
      </div>
    </div>
  );
}
