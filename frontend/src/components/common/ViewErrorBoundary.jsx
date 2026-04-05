import { Component } from "react";

export default class ViewErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { hasError: false, message: "" };
  }

  static getDerivedStateFromError(error) {
    return {
      hasError: true,
      message: error instanceof Error ? error.message : "Unknown frontend render error",
    };
  }

  componentDidCatch(error) {
    console.error("View render failed", error);
  }

  componentDidUpdate(prevProps) {
    if (this.state.hasError && prevProps.resetKey !== this.props.resetKey) {
      this.setState({ hasError: false, message: "" });
    }
  }

  render() {
    if (this.state.hasError) {
      return (
        <section className="sidebar-section">
          <h2>View Recovery</h2>
          <p className="empty-state">
            The page hit a frontend render error, but the app is still running. Refreshing the view or switching tabs
            should recover it.
          </p>
          <small>{this.state.message}</small>
        </section>
      );
    }

    return this.props.children;
  }
}
