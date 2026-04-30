import { SessionProvider } from "./Context/SessionContext";

import {
  BrowserRouter,
  Routes,
  Route,
  Outlet,
  Navigate,
} from "react-router-dom";

import NavBar from "./Assets/Components/NavBar";
import LandingSplash from "./Pages/Landing/LandingSplash";
import RecipeSearchPage from "./Pages/Home/RecipeSearchPage";
import Recipe from "./Pages/Recipe/Recipe";

import "./App.css";

const Layout = () => {
  return (
    <div
      className="app-root"
      style={{ display: "flex", flexDirection: "column", height: "100vh" }}
    >
      <NavBar />
      <Outlet />
    </div>
  );
};

function App() {
  return (
    <SessionProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/" element={<LandingSplash />} />
          <Route element={<Layout />}>
            <Route path="/home" element={<RecipeSearchPage />} />
            <Route path="/recipe/:id" element={<Recipe />} />
          </Route>
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </BrowserRouter>
    </SessionProvider>
  );
}

export default App;
