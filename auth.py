"""
Authentication helpers and route decorators.
"""

from functools import wraps

from flask import session, redirect, url_for, flash, request, g

from database import get_user_by_id


def load_current_user():
    """Load logged-in user into flask.g."""
    user_id = session.get("user_id")
    g.user = get_user_by_id(user_id) if user_id else None


def login_user(user):
    """Store user in session."""
    session["user_id"] = user["id"]
    session["username"] = user["username"]
    session["role"] = user["role"]
    session.permanent = True


def logout_user():
    """Clear session."""
    session.clear()


def login_required(view):
    """Require any logged-in user."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Please log in to continue.", "warning")
            return redirect(url_for("login", next=request.path))
        return view(*args, **kwargs)

    return wrapped


def admin_required(view):
    """Require admin role."""

    @wraps(view)
    def wrapped(*args, **kwargs):
        if not session.get("user_id"):
            flash("Admin login required.", "warning")
            return redirect(url_for("admin_login", next=request.path))
        if session.get("role") != "admin":
            flash("Access denied. Admin account required.", "danger")
            return redirect(url_for("dashboard"))
        return view(*args, **kwargs)

    return wrapped
